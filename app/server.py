#!/usr/bin/env python3
"""MLPrep study app server - stdlib only (Python 3.11).

Serves app/index.html at / and a JSON API over the repo's dsa/ problems,
notes, flashcards, and the practice/queue.tsv spaced-repetition queue.

Run:  PORT=8888 uv run python app/server.py
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
QUEUE = ROOT / "practice" / "queue.tsv"
QUEUE_COLS = ["item_id", "source_path", "added", "last_result", "interval_days", "due"]
QUEUE_CAP = 16  # mirror practice/review.py exactly
_queue_lock = threading.Lock()
LOG = ROOT / "practice" / "log.tsv"
LOG_COLS = ["date", "item_id", "result"]
MAX_BODY = 1_000_000  # POST body cap in bytes

FOLDER_TITLES = {
    "01_arrays_strings_hashing": "Arrays, Strings & Hashing",
    "02_binary_search": "Binary Search",
    "03_stacks_queues_linked_lists": "Stacks, Queues & Linked Lists",
    "04_trees_and_heaps": "Trees & Heaps",
    "05_graphs": "Graphs",
    "06_dp_greedy_backtracking": "DP, Greedy & Backtracking",
    "07_bits_math_matrix": "Bits, Math & Matrix",
}

# Difficulty heuristic for known problem names; everything else is "medium".
EASY = {
    "two_sum", "max_profit", "can_attend_meetings",
    "is_valid", "reverse_list", "reverse_list_recursive", "merge_two_lists",
    "has_cycle", "max_depth", "diameter", "invert", "KthLargest",
    "climb_stairs", "single_number", "hamming_weight", "count_bits",
    "missing_number_xor", "missing_number_sum", "reverse_bits", "is_happy",
    "is_palindrome_number",
}
HARD = {
    "trap", "min_window", "max_sliding_window", "find_median_sorted_arrays",
    "largest_rectangle_area", "serialize", "merge_k_lists", "MedianFinder",
    "find_words", "ladder_length", "n_queens",
}
TIME_BOX = {"easy": 15, "medium": 25, "hard": 35}

# Names that are never problems even with a docstring (shared scaffolding).
SKIP_NAMES = {"ListNode", "TreeNode", "Node", "GraphNode"}
MIN_DOC_LEN = 80  # a real problem statement is longer than a helper's one-liner


# ---------------------------------------------------------------------------
# Problem discovery (ast-based, cached on file mtime)
# ---------------------------------------------------------------------------
_cache = {}  # str(path) -> (mtime, data)


def _difficulty(name):
    if name in EASY:
        return "easy"
    if name in HARD:
        return "hard"
    return "medium"


def _split_doc(doc):
    """Split a docstring into (statement, key_insight).

    In this repo the docstrings are: problem statement paragraph(s), then a
    paragraph starting with "Pattern:" / "Key insight:" / "Complexity:" that
    names the pattern, the insight, and the complexity. That tail is the hint,
    not the statement - showing it up front would spoil the problem.
    """
    HINT_MARKERS = ("pattern", "key insight", "complexity", "state:",
                    "invariant:", "approach:", "hint:", "recurrence:",
                    "decision:", "why it works", "backtracking", "greedy",
                    "dp:", "dp ", "monotonic", "two pointers", "two-heap",
                    "sliding window", "binary search on", "bfs", "dfs",
                    "union-find", "dijkstra", "kadane", "xor ", "bitmask",
                    "heap:", "space optimization")
    paras = re.split(r"\n\s*\n", doc.strip())
    stmt, insight = [], []
    for p in paras:
        head = p.strip().lower()
        if insight or (stmt and head.startswith(HINT_MARKERS)):
            insight.append(p)
        else:
            stmt.append(p)
    return "\n\n".join(stmt), "\n\n".join(insight)


def _statement_md(stmt):
    """Format a raw docstring statement for markdown rendering. Docstrings hold
    Example lines as plain indented text; markdown collapses single newlines,
    smearing them into one run-on line. Examples become fenced code blocks and
    Note/Constraints paragraphs become blockquote callouts, giving the three
    kinds of information (task, examples, rules) distinct visual forms."""
    import textwrap

    def looks_code(para):
        lines = [l for l in para.split("\n") if l.strip()]
        indented = sum(1 for l in lines if l.startswith("  "))
        return lines and (indented >= len(lines) / 2
                          or lines[0].lstrip().startswith(("Input:", "Output:")))

    def fence(lines):
        body = textwrap.dedent("\n".join(l.rstrip() for l in lines)).strip("\n")
        return f"\n\n```text\n{body}\n```" if body else ""

    paras = re.split(r"\n\s*\n", stmt.strip())
    out, i = [], 0
    while i < len(paras):
        para = paras[i]
        first, *rest = para.split("\n")
        head = first.strip().lower()
        if head.startswith("example"):
            label, _, tail = first.strip().partition(":")
            body_lines = ([tail] if tail.strip() else []) + rest
            # absorb following code-like paragraphs (blank-line-after-header
            # style), keeping their blank-line separation inside the fence
            while i + 1 < len(paras) and looks_code(paras[i + 1]):
                body_lines += [""] + paras[i + 1].split("\n")
                i += 1
            out.append(f"**{label}:**" + fence(body_lines))
        elif head.startswith(("note:", "constraints:")):
            out.append("> " + para.strip().replace("\n", "\n> "))
        elif looks_code(para):  # orphan indented block - fence it verbatim
            out.append(fence(para.split("\n")).lstrip("\n"))
        else:
            out.append(para)
        i += 1
    return "\n\n".join(out)


def _first_sentence(text):
    """First sentence of the statement, unwrapped, for table subtitles."""
    flat = " ".join(text.strip().split())
    m = re.match(r"(.+?\.)(?:\s|$)", flat)
    s = m.group(1) if m else flat
    return s if len(s) <= 140 else s[:137] + "..."


def _doc_block(doc, ind):
    lines = doc.split("\n")
    if len(lines) == 1:
        return [f'{ind}"""{doc}"""']
    out = [f'{ind}"""{lines[0]}']
    out += [f"{ind}{l}" if l.strip() else "" for l in lines[1:]]
    out.append(f'{ind}"""')
    return out


def _fn_stub_lines(node, doc, indent=""):
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    out = [f"{indent}def {node.name}({args}){ret}:"]
    if doc:
        out += _doc_block(doc, indent + "    ")
    out.append(f"{indent}    pass")
    return out


def _make_stub(node, statement):
    """Signature + statement docstring + pass - the solution body is dropped."""
    if isinstance(node, ast.ClassDef):
        bases = ""
        if node.bases:
            bases = "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")"
        out = [f"class {node.name}{bases}:"]
        out += _doc_block(statement, "    ")
        for item in node.body:
            # public methods + dunders define the required interface; private
            # helpers (_unlink, ...) and inner classes are implementation
            # choices the user makes on their own.
            if isinstance(item, ast.FunctionDef) and (
                    not item.name.startswith("_") or item.name.startswith("__")):
                out.append("")
                mdoc = ast.get_docstring(item) or ""
                out += _fn_stub_lines(item, mdoc, "    ")
        return "\n".join(out) + "\n"
    return "\n".join(_fn_stub_lines(node, statement)) + "\n"


def _parse_problems_file(path):
    mtime = path.stat().st_mtime
    hit = _cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]

    src = path.read_text()
    tree = ast.parse(src)
    folder = path.parent.name
    problems, helpers = [], {}

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node) or ""
        seg = ast.get_source_segment(src, node) or ""
        is_problem = (len(doc) > MIN_DOC_LEN and not node.name.startswith("_")
                      and node.name not in SKIP_NAMES)
        if not is_problem:
            helpers[node.name] = seg
            continue
        statement, insight = _split_doc(doc)
        if not statement:  # docstring opens with "Key insight" - keep 1st para
            statement, insight = doc.split("\n\n")[0], insight or doc
        statement_md = _statement_md(statement)
        problems.append({
            "id": f"{folder}/{node.name}",
            "name": node.name,
            "kind": "class" if isinstance(node, ast.ClassDef) else "function",
            "folder": folder,
            "folder_title": FOLDER_TITLES.get(folder, folder),
            "statement": statement_md,
            "statement_first_line": _first_sentence(statement),
            "key_insight": insight,
            "stub": _make_stub(node, statement),
            "solution_source": seg,
            "difficulty": _difficulty(node.name),
            "time_box_min": TIME_BOX[_difficulty(node.name)],
            "source_path": str(path.relative_to(ROOT)),
        })

    # attach helper sources referenced (transitively) by each problem
    for p in problems:
        wanted, frontier = [], [p["solution_source"]]
        while frontier:
            text = frontier.pop()
            for name, hsrc in helpers.items():
                if name not in wanted and re.search(rf"\b{re.escape(name)}\b", text):
                    wanted.append(name)
                    frontier.append(hsrc)
        p["helpers"] = "\n\n\n".join(helpers[n] for n in wanted)

    data = {"problems": problems,
            "assert_count": sum(isinstance(n, ast.Assert) for n in ast.walk(tree))}
    _cache[str(path)] = (mtime, data)
    return data


def _problem_files():
    return sorted((ROOT / "dsa").glob("*/problems.py"))


def all_problems():
    out = []
    for f in _problem_files():
        try:
            out.extend(_parse_problems_file(f)["problems"])
        except Exception as e:  # a broken problems.py must not take down the rest
            print(f"warning: skipping {f}: {type(e).__name__}: {e}", file=sys.stderr)
    return out


def find_problem(pid):
    for p in all_problems():
        if p["id"] == pid:
            return p
    return None


# ---------------------------------------------------------------------------
# Queue (mirrors practice/review.py's file format and mechanics exactly)
# ---------------------------------------------------------------------------
_skipped_rows = 0  # malformed queue.tsv rows seen on the most recent load


def queue_load():
    global _skipped_rows
    if not QUEUE.exists():
        QUEUE.write_text("\t".join(QUEUE_COLS) + "\n")
    lines = QUEUE.read_text().splitlines()[1:]
    rows, skipped = [], 0
    for ln in lines:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) != len(QUEUE_COLS):
            skipped += 1
            continue
        rows.append(dict(zip(QUEUE_COLS, parts)))
    if skipped:
        print(f"warning: {QUEUE.name}: skipped {skipped} malformed row(s)",
              file=sys.stderr)
    _skipped_rows = skipped
    return rows


def queue_save(rows):
    """Atomic write: temp file in the same directory, then os.replace."""
    out = ["\t".join(QUEUE_COLS)] + ["\t".join(r[c] for c in QUEUE_COLS) for r in rows]
    fd, tmp = tempfile.mkstemp(dir=QUEUE.parent, prefix=".queue_", suffix=".tsv")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(out) + "\n")
        os.replace(tmp, QUEUE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _log_rows():
    """Parsed practice/log.tsv history (append-only; header skipped)."""
    if not LOG.exists():
        return []
    rows = []
    for ln in LOG.read_text().splitlines()[1:]:
        parts = ln.split("\t")
        if len(parts) == len(LOG_COLS):
            rows.append(dict(zip(LOG_COLS, parts)))
    return rows


def _log_append(item_id, result):
    """Append one history line. Caller must hold _queue_lock."""
    if not LOG.exists():
        LOG.write_text("\t".join(LOG_COLS) + "\n")
    with LOG.open("a") as fh:
        fh.write(f"{date.today().isoformat()}\t{item_id}\t{result}\n")


def queue_log(item_id, source_path, result):
    """Misses enter the queue (interval 1); existing rows: pass doubles
    (cap 16) / fail resets. A pass on an unqueued item is a no-op - only
    misses (and solution peeks) need spaced review. Same-day duplicate
    results on an existing row are also no-ops (double-click protection).
    Every applied result is appended to practice/log.tsv."""
    with _queue_lock:
        rows = queue_load()
        row = next((r for r in rows if r["item_id"] == item_id), None)
        today = date.today()
        if row is None:
            if result == "pass":
                return {"logged": False,
                        "reason": "not queued - only misses (and solution "
                                  "peeks) enter the review queue"}
            row = {"item_id": item_id, "source_path": source_path or "",
                   "added": today.isoformat(), "last_result": "new",
                   "interval_days": "1",
                   "due": (today + timedelta(days=1)).isoformat()}
            rows.append(row)
        else:
            t = today.isoformat()
            if any(l["date"] == t and l["item_id"] == item_id
                   and l["result"] == result for l in _log_rows()):
                return {"logged": False, "reason": "already logged today",
                        "row": row}
        iv = min(int(row["interval_days"]) * 2, QUEUE_CAP) if result == "pass" else 1
        row.update(last_result=result, interval_days=str(iv),
                   due=(today + timedelta(days=iv)).isoformat())
        queue_save(rows)
        _log_append(item_id, result)
        return {"logged": True, "row": row}


def queue_snapshot():
    rows = queue_load()
    today = date.today()
    due = sorted((r for r in rows if r["due"] <= today.isoformat()),
                 key=lambda r: (r["due"], r["item_id"]))
    buckets = {1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    for r in rows:
        iv = int(r["interval_days"])
        buckets[min((b for b in buckets if iv <= b), default=QUEUE_CAP)] += 1
    reviewed_today = 0
    for r in rows:
        try:  # marking sets due = today + interval, so invert that
            if date.fromisoformat(r["due"]) - timedelta(days=int(r["interval_days"])) == today:
                reviewed_today += 1
        except ValueError:
            pass
    return {"rows": rows, "due_today": due, "due_count": len(due),
            "today": today.isoformat(),
            "histogram": [{"interval": k, "count": v} for k, v in sorted(buckets.items())],
            "reviewed_today": reviewed_today, "skipped_rows": _skipped_rows}


def activity():
    """History rollup from log.tsv: per-day counts for the last 120 calendar
    days that have entries, plus today's reviewed/due tallies."""
    by_date = {}
    for l in _log_rows():
        d = by_date.setdefault(l["date"], {"date": l["date"], "count": 0,
                                           "passes": 0, "fails": 0})
        d["count"] += 1
        if l["result"] == "pass":
            d["passes"] += 1
        elif l["result"] == "fail":
            d["fails"] += 1
    today = date.today()
    cutoff = (today - timedelta(days=119)).isoformat()
    days = sorted((v for k, v in by_date.items() if k >= cutoff),
                  key=lambda v: v["date"])
    reviewed = by_date.get(today.isoformat(), {}).get("count", 0)
    due = queue_snapshot()["due_count"]
    return {"days": days, "today": {"reviewed": reviewed, "due": due}}


def problem_status(pid, queue_rows):
    row = next((r for r in queue_rows if r["item_id"] == f"app:{pid}"), None)
    if row is None:
        return "todo"
    return "solved" if row["last_result"] == "pass" else "attempted"


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------
def _module_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.Import):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names |= {a.asname or a.name for a in n.names}
    return names


def run_judge(pid, code):
    """AST-swap the user's def into the original problems.py and run it.

    NOTE on semantics: the original file's __main__ block asserts EVERY problem
    in that file, so the reference solutions for the other problems execute too
    - only the target def is replaced by the user's code. ok:true therefore
    means the whole suite (including every assert for the user's function)
    passed with the user's implementation in place.
    """
    prob = find_problem(pid)
    if prob is None:
        return {"ok": False, "error": f"unknown problem id: {pid}",
                "output": "", "elapsed_ms": 0}, 404
    folder, name = pid.split("/", 1)
    path = ROOT / "dsa" / folder / "problems.py"
    orig_src = path.read_text()
    orig = ast.parse(orig_src)

    try:
        user = ast.parse(code)
    except SyntaxError as e:
        msg = f"SyntaxError: {e.msg} (line {e.lineno})"
        if e.text:
            msg += "\n    " + e.text.strip()
        return {"ok": False, "error": msg, "output": "", "elapsed_ms": 0}, 200

    target, extras = None, []
    for n in user.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and n.name == name:
            target = n
        else:
            extras.append(n)
    if target is None:
        return {"ok": False, "output": "", "elapsed_ms": 0,
                "error": f"your code must define {name!r} at the top level "
                         f"(a {prob['kind']} of that exact name)"}, 200

    # keep the user's extra top-level defs/imports/constants that don't
    # collide with names the original module already binds
    existing = _module_names(orig)
    keep = []
    for n in extras:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name not in existing:
                keep.append(n)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound = {(a.asname or a.name).split(".")[0] for a in n.names}
            if not (bound & existing):
                keep.append(n)
        elif isinstance(n, ast.Assign):
            tgt, simple = set(), True
            for t in n.targets:
                if isinstance(t, ast.Name):
                    tgt.add(t.id)
                elif isinstance(t, ast.Tuple) and \
                        all(isinstance(e, ast.Name) for e in t.elts):
                    tgt |= {e.id for e in t.elts}
                else:
                    simple = False
            if simple and tgt and not (tgt & existing):
                keep.append(n)
        elif isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name) and n.target.id not in existing:
                keep.append(n)
        # anything else (docstrings, __main__ blocks, prints) is dropped

    new_body = []
    for n in orig.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            new_body.extend(keep)
            new_body.append(target)
        else:
            new_body.append(n)
    module = ast.fix_missing_locations(ast.Module(body=new_body, type_ignores=[]))
    merged_src = ast.unparse(module)
    assert_count = _parse_problems_file(path)["assert_count"]

    # temp file lives in the SAME directory as the original so any relative
    # behavior (imports, paths) matches; deleted after the run
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="_judge_", suffix=".py")
    t0 = time.time()
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(merged_src)
        try:
            proc = subprocess.run([sys.executable, tmp], cwd=ROOT,
                                  capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": "",
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "error": "Timed out after 30 s - likely an infinite loop.",
                    "asserts": assert_count}, 200
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    elapsed = int((time.time() - t0) * 1000)
    pretty = f"dsa/{folder}/problems.py (judge copy)"
    stderr = proc.stderr.replace(tmp, pretty)
    ok = proc.returncode == 0
    error = ""
    if not ok:
        # pull the failing line of the merged source out of the traceback
        lines = re.findall(rf'File "{re.escape(pretty)}", line (\d+)', stderr)
        verdict = ""
        if lines:
            lineno = int(lines[-1])
            src_lines = merged_src.split("\n")
            if 0 < lineno <= len(src_lines):
                verdict = f"Failing line {lineno}:  {src_lines[lineno - 1].strip()}"
        tail = stderr.strip().split("\n")[-1] if stderr.strip() else "non-zero exit"
        error = "\n".join(x for x in (tail, verdict, "", stderr.strip()) if x is not None)
    return {"ok": ok, "output": proc.stdout, "error": error,
            "elapsed_ms": elapsed, "asserts": assert_count}, 200


# ---------------------------------------------------------------------------
# Playground: edit-and-run any curriculum script (ml/ and dsa/ only)
# ---------------------------------------------------------------------------
def _lab_path(raw_path):
    """Resolve a playground-eligible script: a .py inside ml/ or dsa/."""
    p = (ROOT / raw_path).resolve()
    ok = (p.is_relative_to(ROOT) and p.suffix == ".py"
          and p.relative_to(ROOT).parts[0] in ("ml", "dsa"))
    if not ok:
        return None, (400, "path must be a .py file under ml/ or dsa/")
    if not p.is_file():
        return None, (404, "no such file")
    return p, None


def _nb_path(raw_path):
    """Resolve a notebook-viewable file: a .py or .ipynb inside ml/ or dsa/.
    The playground (whole-file run) stays .py-only via _lab_path; the notebook
    view and kernel accept both."""
    p = (ROOT / raw_path).resolve()
    ok = (p.is_relative_to(ROOT) and p.suffix in (".py", ".ipynb")
          and p.relative_to(ROOT).parts[0] in ("ml", "dsa"))
    if not ok:
        return None, (400, "path must be a .py or .ipynb file under ml/ or dsa/")
    if not p.is_file():
        return None, (404, "no such file")
    return p, None


def run_lab(raw_path, code):
    """Run an edited copy of a curriculum script. The edit is written to a
    scratch file in the script's own directory (so imports/paths behave) and
    deleted after; the repo file is never touched. The file's own asserts and
    printed checks are the feedback - exit 0 means every check passed."""
    path, err = _lab_path(raw_path)
    if err:
        return {"ok": False, "output": "", "error": err[1], "elapsed_ms": 0}, err[0]
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="_lab_", suffix=".py")
    t0 = time.time()
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(code)
        try:
            proc = subprocess.run([sys.executable, tmp], cwd=ROOT,
                                  capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": "",
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "error": "Timed out after 180 s - infinite loop, or a "
                             "training script grown past playground size."}, 200
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    elapsed = int((time.time() - t0) * 1000)
    pretty = str(path.relative_to(ROOT)) + " (edited copy)"
    stderr = proc.stderr.replace(tmp, pretty)
    error = ""
    if proc.returncode != 0:
        lines = re.findall(rf'File "{re.escape(pretty)}", line (\d+)', stderr)
        verdict = ""
        if lines:
            lineno = int(lines[-1])
            src = code.split("\n")
            if 0 < lineno <= len(src):
                verdict = f"Failing line {lineno}:  {src[lineno - 1].strip()}"
        tail = stderr.strip().split("\n")[-1] if stderr.strip() else "non-zero exit"
        error = "\n".join(x for x in (tail, verdict, "", stderr.strip()) if x)
    return {"ok": proc.returncode == 0, "output": proc.stdout,
            "error": error, "elapsed_ms": elapsed}, 200


# ---------------------------------------------------------------------------
# Notebook kernels: cell-based execution with a persistent worker process
# ---------------------------------------------------------------------------
NB_TIMEOUT = float(os.environ.get("NB_TIMEOUT", "120"))  # seconds per cell
NB_IDLE_SEC = 30 * 60  # reap sessions idle longer than this
NB_MAX_SESSIONS = 4    # LRU-kill beyond this many live kernels
NB_CELL_CAP = 80       # soft max lines per heuristic cell

# The worker: a bare exec loop over one shared namespace, so state (imports,
# defs, arrays) persists across cells - the Jupyter paradigm. Protocol on
# stdin/stdout: 4-byte big-endian length + JSON in both directions.
NB_WORKER_SRC = r'''
import contextlib, io, json, struct, sys, time, traceback
try:
    import matplotlib
    matplotlib.use("Agg")  # never let a stray import hang on a GUI backend
except Exception:
    pass
_ns = {"__name__": "__main__"}
if len(sys.argv) > 1:
    _ns["__file__"] = sys.argv[1]
_in, _out = sys.stdin.buffer, sys.stdout.buffer


def _read():
    head = _in.read(4)
    if len(head) < 4:
        return None
    n = struct.unpack(">I", head)[0]
    data = b""
    while len(data) < n:
        chunk = _in.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return json.loads(data)


def _write(obj):
    data = json.dumps(obj).encode()
    _out.write(struct.pack(">I", len(data)) + data)
    _out.flush()


while True:
    msg = _read()
    if msg is None:
        break
    out, err = io.StringIO(), io.StringIO()
    t0 = time.time()
    ok = True
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(msg.get("code", ""), "<cell>", "exec"), _ns)
    except BaseException:
        ok = False
        err.write(traceback.format_exc())
    _write({"ok": ok, "stdout": out.getvalue(), "stderr": err.getvalue(),
            "elapsed_ms": int((time.time() - t0) * 1000)})
'''

_nb_sessions = {}  # session_id -> {id, path, proc, lock, last_use, dead}
_nb_lock = threading.Lock()  # guards the sessions dict itself


def _nb_spawn(path):
    """One kernel process; cwd is the file's directory so imports and
    relative paths behave exactly like running the file there."""
    return subprocess.Popen(
        [sys.executable, "-u", "-c", NB_WORKER_SRC, str(path)],
        cwd=path.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)


def _nb_kill_proc(proc):
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def _nb_reap():
    """Lazily drop idle sessions; called on every nb endpoint hit."""
    now = time.time()
    with _nb_lock:
        for sid in [s for s, v in _nb_sessions.items()
                    if now - v["last_use"] > NB_IDLE_SEC]:
            _nb_kill_proc(_nb_sessions.pop(sid)["proc"])


def nb_start(raw_path):
    _nb_reap()
    path, err = _nb_path(raw_path)
    if err:
        return {"error": err[1]}, err[0]
    sid = os.urandom(16).hex()
    sess = {"id": sid, "path": path, "proc": _nb_spawn(path),
            "lock": threading.Lock(), "last_use": time.time(), "dead": False}
    with _nb_lock:
        _nb_sessions[sid] = sess
        while len(_nb_sessions) > NB_MAX_SESSIONS:  # LRU-kill the oldest
            oldest = min(_nb_sessions, key=lambda s: _nb_sessions[s]["last_use"])
            if oldest == sid:
                break
            _nb_kill_proc(_nb_sessions.pop(oldest)["proc"])
    return {"session": sid}, 200


def _nb_exec(proc, code):
    """Ship one cell to the worker and wait for its reply.

    Returns (reply_or_None, timed_out). The blocking pipe I/O runs on a helper
    thread so the timeout can fire; on timeout the process is killed, which
    also unblocks the reader with EOF.
    """
    payload = json.dumps({"code": code}).encode()
    box = {}

    def talk():
        try:
            proc.stdin.write(len(payload).to_bytes(4, "big") + payload)
            proc.stdin.flush()
            head = proc.stdout.read(4)
            if len(head) < 4:
                return
            n = int.from_bytes(head, "big")
            data = b""
            while len(data) < n:
                chunk = proc.stdout.read(n - len(data))
                if not chunk:
                    return
                data += chunk
            box["msg"] = json.loads(data)
        except Exception:
            pass

    t = threading.Thread(target=talk, daemon=True)
    t.start()
    t.join(NB_TIMEOUT)
    if t.is_alive():
        _nb_kill_proc(proc)
        t.join(2)
        return None, True
    return box.get("msg"), False


def nb_run(sid, code):
    _nb_reap()
    with _nb_lock:
        sess = _nb_sessions.get(sid)
    if sess is None:
        return {"ok": False, "dead": True,
                "error": "unknown session (expired or evicted): start a "
                         "fresh kernel"}, 404
    if not sess["lock"].acquire(blocking=False):
        return {"error": "kernel busy"}, 409
    try:
        sess["last_use"] = time.time()
        proc = sess["proc"]
        if sess["dead"] or proc.poll() is not None:
            sess["dead"] = True
            return {"ok": False, "dead": True,
                    "error": "kernel is dead: restart it and run cells "
                             "again"}, 200
        msg, timed_out = _nb_exec(proc, code)
        sess["last_use"] = time.time()
        if timed_out:
            if sess["proc"] is proc:
                sess["dead"] = True
            return {"ok": False, "dead": True, "stdout": "", "stderr": "",
                    "elapsed_ms": int(NB_TIMEOUT * 1000),
                    "error": f"cell timed out after {NB_TIMEOUT:g} s, kernel "
                             "restarted: run cells again"}, 200
        if msg is None:  # worker crashed or was killed mid-cell
            if sess["proc"] is proc:
                sess["dead"] = True
            return {"ok": False, "dead": True,
                    "error": "kernel process died mid-cell: restart and run "
                             "cells again"}, 200
        return msg, 200
    finally:
        sess["lock"].release()


def nb_restart(sid):
    _nb_reap()
    with _nb_lock:
        sess = _nb_sessions.get(sid)
    if sess is None:
        return {"error": "unknown session"}, 404
    _nb_kill_proc(sess["proc"])  # unblocks any in-flight run with EOF
    with sess["lock"]:
        sess["proc"] = _nb_spawn(sess["path"])
        sess["dead"] = False
        sess["last_use"] = time.time()
    return {"session": sid, "restarted": True}, 200


def nb_stop(sid):
    with _nb_lock:
        sess = _nb_sessions.pop(sid, None)
    if sess is not None:
        _nb_kill_proc(sess["proc"])
    return {"stopped": sess is not None}, 200


def _split_cells(src):
    """Split a script into notebook cells, newline-preserving: the cells
    always concatenate back to the original file byte-for-byte (asserted).

    Explicit "# %%" separator lines win if present. Otherwise heuristic:
    new cell at every top-level banner comment block (# --- / # ===), at
    every top-level def/class that follows non-def code, and at the
    __main__ block. Oversized cells are split at top-level statement
    boundaries via ast, never inside a def/class/try/with body.
    """
    lines = src.splitlines(keepends=True)
    if not lines:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [src]

    def start0(node):  # 0-based first line, decorators included
        lo = node.lineno
        for d in getattr(node, "decorator_list", None) or []:
            lo = min(lo, d.lineno)
        return lo - 1

    spans = [(start0(nd), (nd.end_lineno or nd.lineno) - 1) for nd in tree.body]

    def inside_stmt(i):  # line sits inside a top-level statement's span
        return any(a <= i <= b for a, b in spans)

    def is_comment(i):  # top-level (column-0) comment line
        return lines[i].startswith("#") and not inside_stmt(i)

    def attach(i):  # pull a boundary up over comment lines glued right above
        while i > 0 and is_comment(i - 1):
            i -= 1
        return i

    explicit = {i for i, l in enumerate(lines) if re.match(r"# %%", l)}
    if explicit:
        bounds = {0} | explicit
    else:
        banner_re = re.compile(r"#\s*[-=]{3,}")
        bounds, banners = {0}, set()
        for i, l in enumerate(lines):
            # first line of a banner block only, so the block stays whole
            if (banner_re.match(l) and not inside_stmt(i)
                    and (i == 0 or not is_comment(i - 1))):
                bounds.add(i)
                banners.add(i)
        prev = None
        DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        for nd in tree.body:
            s0 = start0(nd)
            is_main = (isinstance(nd, ast.If)
                       and re.match(r"if\s+__name__\b", lines[s0]))
            wants = is_main or (isinstance(nd, DEFS) and prev is not None
                                and not isinstance(prev, DEFS))
            # a banner in the gap above already starts this cell: don't cut
            # the banner off into a tiny cell of its own
            if wants and not (prev is not None and
                              any(prev.end_lineno <= j < s0 for j in banners)):
                bounds.add(attach(s0))
            prev = nd

    # cap pass: split any oversized cell at top-level statement starts
    ordered = sorted(bounds)
    out = []
    for a, b in zip(ordered, ordered[1:] + [len(lines)]):
        out.append(a)
        if b - a <= NB_CELL_CAP:
            continue
        cur = a
        for nd in tree.body:
            s0, e0 = start0(nd), (nd.end_lineno or nd.lineno) - 1
            if not (a < s0 and e0 < b):
                continue
            w = attach(s0)
            if w > cur and e0 - cur + 1 > NB_CELL_CAP:
                out.append(w)
                cur = w
        # a single statement longer than the cap stays whole: never split
        # inside a def/class/try/with body

    ordered = sorted(set(out))
    cells = ["".join(lines[a:b])
             for a, b in zip(ordered, ordered[1:] + [len(lines)])]
    cells = [c for c in cells if c]
    assert "".join(cells) == src, "cell split lost bytes"
    return cells


_BANNER_LINE = re.compile(r"#\s*[-=]{3,}")


def _cell_kind(cell_src):
    """"prose" when the cell holds no executable code: only comments, blank
    lines, and/or bare string-literal expressions (the module docstring)."""
    try:
        tree = ast.parse(cell_src)
    except SyntaxError:
        return "code"
    for n in tree.body:
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)):
            return "code"
    return "prose"


def _prose_pieces(cell, first):
    """Split one cell into [(kind, source)] pieces, byte-exact.

    A cell that is entirely comments and/or a string-literal expression is
    prose. A code cell that BEGINS with the module docstring (first cell of
    the file only) or with a banner comment block (>= 2 leading #-comment
    lines, or any leading #---/#=== rule line) gets that prefix split off as
    its own prose cell; blank separator lines travel with the prose block so
    the code cell starts clean."""
    if _cell_kind(cell) == "prose":
        return [("prose", cell)]
    lines = cell.splitlines(keepends=True)
    cut = 0
    if first:  # module docstring prefix (any shebang/comment lines above it)
        try:
            body = ast.parse(cell).body
        except SyntaxError:
            body = []
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            cut = body[0].end_lineno  # 1-based end == 0-based next line
    if not cut:  # leading banner comment block
        n = 0
        while n < len(lines) and lines[n].startswith("#"):
            n += 1
        if n >= 2 or any(_BANNER_LINE.match(l) for l in lines[:n]):
            cut = n
    if cut:
        while cut < len(lines) and not lines[cut].strip():
            cut += 1
        head, rest = "".join(lines[:cut]), "".join(lines[cut:])
        if head and rest:
            return [("prose", head)] + _prose_pieces(rest, False)
    return [("code", cell)]


def _ipynb_cells(path):
    """[(kind, source)] pieces of a .ipynb (nbformat v4) file: markdown cells
    are prose, code cells are code, anything else (raw) is skipped. Source
    lists are joined; no nbformat dependency needed - it's just JSON."""
    nb = json.loads(path.read_text())
    out = []
    for c in nb.get("cells", []):
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        kind = {"markdown": "prose", "code": "code"}.get(c.get("cell_type"))
        if kind:
            out.append((kind, src))
    return out


def nb_cells(raw_path):
    _nb_reap()
    path, err = _nb_path(raw_path)
    if err:
        return {"error": err[1]}, err[0]
    if path.suffix == ".ipynb":
        pieces = _ipynb_cells(path)
        title, fmt = _ipynb_meta(path)[0], "ipynb"
    else:
        src = path.read_text()
        pieces, first = [], True
        for chunk in _split_cells(src):
            pieces.extend(_prose_pieces(chunk, first))
            first = False
        assert "".join(s for _, s in pieces) == src, "prose split lost bytes"
        title, fmt = _script_meta(path)[0], "py"
    d = path.parent
    notes = next((str((d / n).relative_to(ROOT))
                  for n in ("notes.md", "README.md") if (d / n).is_file()), None)
    return {"path": raw_path, "file": path.name, "format": fmt,
            "title": title, "notes": notes,
            "cells": [{"kind": k, "source": s} for k, s in pieces]}, 200


# ---------------------------------------------------------------------------
# Notebook index: docstring-derived titles + curriculum order
# ---------------------------------------------------------------------------
ML_FOLDER_TITLES = {
    "01_foundations": "Foundations",
    "02_classical_ml": "Classical ML",
    "03_neural_networks": "Neural Networks",
    "04_deep_learning": "Deep Learning",
    "05_evaluation": "Evaluation",
    "06_ml_systems": "ML Systems",
    "07_llm_stack": "LLM Stack",
    "08_interview_drills": "Interview Drills",
    "09_reinforcement_learning": "Reinforcement Learning",
}

# Pedagogical order of the notebooks inside each module, matched to the flow
# of that module's notes.md section headers. Anything unlisted (including the
# single-script modules and every dsa problems.py) appends alphabetically.
NB_ORDER = {
    # notes: Part 2 calculus/chain rule before Part 3 probability
    "01_foundations": ["gradients_from_scratch.py", "probability_demos.py"],
    # notes: linear/logistic (SS1-2) -> trees/ensembles (SS5-6) ->
    # k-means/PCA (SS8-9) -> GMM/EM (SS11) -> matrix factorization (SS13)
    "02_classical_ml": ["linear_models_scratch.py",
                        "trees_and_ensembles_scratch.py",
                        "kmeans_pca_scratch.py", "gmm_em_scratch.py",
                        "matrix_factorization_scratch.py"],
    # notes: backprop/chain rule (b) -> losses+optimizers vectorized (c, f) ->
    # init/lr/vanishing-gradient experiments (d, e, f folklore)
    "03_neural_networks": ["micrograd_plus.py", "mlp_numpy.py",
                           "training_dynamics.py"],
    # notes: CNNs (SS1) -> attention (SS3) -> language modeling/GPT (SS3.6, SS5)
    "04_deep_learning": ["cnn_scratch.py", "attention_scratch.py",
                         "mini_gpt.py"],
    # notes: confusion matrix/ROC (SS1-2) -> calibration (SS3) ->
    # bias-variance + cross-validation (SS4-5)
    "05_evaluation": ["metrics_scratch.py", "calibration_scratch.py",
                      "bias_variance_experiment.py"],
    # notes: training memory arithmetic (SS2) -> LoRA fine-tuning (SS4) ->
    # speculative decoding inference (SS5)
    "07_llm_stack": ["memory_math.py", "lora_scratch.py",
                     "speculative_decoding_demo.py"],
    "09_reinforcement_learning": ["rl_scratch.py"],
}


def _folder_title(track, name):
    if track == "dsa" and name in FOLDER_TITLES:
        return FOLDER_TITLES[name]
    if name in ML_FOLDER_TITLES:
        return ML_FOLDER_TITLES[name]
    return re.sub(r"^\d+_", "", name).replace("_", " ").title()


_meta_cache = {}  # str(path) -> (mtime, (title, blurb))


def _script_meta(path):
    """(title, blurb) for a .py script, derived from its module docstring and
    cached on mtime. Title = the docstring's first sentence (wrapped lines
    rejoined, any leading "filename.py -" prefix dropped, trailing period
    stripped; fallback: humanized filename). Blurb = the next non-empty
    docstring line's sentence, capped at 120 chars."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    hit = _meta_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    doc = ""
    try:
        doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    except (SyntaxError, OSError, ValueError):
        pass
    lines = doc.split("\n")

    def take_flat(i):  # rejoin wrapped lines until a sentence end / blank line
        buf = []
        while i < len(lines):
            l = lines[i].strip()
            if not l:
                break
            buf.append(l)
            i += 1
            if re.search(r"[.!?]$", l):
                break
        return " ".join(buf), i

    def first_sentence(flat):
        m = re.match(r"(.+?[.!?])(?:\s|$)", flat)
        return (m.group(1) if m else flat).strip()

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    flat, i = take_flat(i)
    flat = re.sub(rf"^{re.escape(path.name)}\s*[-:]+\s*", "", flat)
    title = first_sentence(flat).rstrip(".:").strip()
    if not title:
        title = path.stem.replace("_", " ")
    title = title[:1].upper() + title[1:]
    if len(title) > 120:  # a run-on first sentence: cut at its colon if any
        cut = title.find(": ")
        title = title[:cut] if cut >= 20 else title[:117].rstrip() + "..."
    while True:  # blurb start: skip blanks, Run: lines, and label-only lines
        while i < len(lines) and (not lines[i].strip()
                                  or lines[i].strip().startswith("Run:")):
            i += 1
        l = lines[i].strip() if i < len(lines) else ""
        if l.endswith(":") and len(l) <= 40:  # "What this script does:" etc
            i += 1
            continue
        break
    bflat = re.sub(r"^\d+[.)]\s+", "", take_flat(i)[0])  # drop "1. " markers
    blurb = first_sentence(bflat).rstrip(":")
    if len(blurb) > 120:
        blurb = blurb[:117].rstrip() + "..."
    meta = (title, blurb)
    _meta_cache[str(path)] = (mtime, meta)
    return meta


_ipynb_meta_cache = {}  # str(path) -> (mtime, (title, blurb))


def _ipynb_meta(path):
    """(title, blurb) for a .ipynb notebook, cached on mtime like _script_meta.
    Title = the first markdown cell's H1 line stripped of leading #'s
    (fallback: humanized stem). Blurb = the first non-heading paragraph line
    of that same cell, capped at 120 chars."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    hit = _ipynb_meta_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    title, blurb = "", ""
    try:
        cell = next((s for k, s in _ipynb_cells(path) if k == "prose"), "")
        for line in cell.splitlines():
            l = line.strip()
            if not l:
                continue
            if l.startswith("#"):
                if not title:
                    title = l.lstrip("#").strip()
                continue
            blurb = l
            break
    except (OSError, ValueError):
        pass
    if not title:
        title = path.stem.replace("_", " ")
        title = title[:1].upper() + title[1:]
    if len(blurb) > 120:
        blurb = blurb[:117].rstrip() + "..."
    meta = (title, blurb)
    _ipynb_meta_cache[str(path)] = (mtime, meta)
    return meta


def _file_meta(path):
    """(title, blurb) for either notebook flavor."""
    return _ipynb_meta(path) if path.suffix == ".ipynb" else _script_meta(path)


def _notebook_files(d):
    """The folder's notebook files (.py + .ipynb), with a .py hidden whenever
    a same-stem .ipynb exists (the transition state while scripts convert)."""
    files = list(d.glob("*.py")) + list(d.glob("*.ipynb"))
    nb_stems = {f.stem for f in files if f.suffix == ".ipynb"}
    return [f for f in files
            if not (f.suffix == ".py" and f.stem in nb_stems)]


def notebooks_index():
    """Modules with notebooks (.py or .ipynb), in curriculum order (ml then
    dsa, folders numbered), each with its notes file and its notebooks in
    NB_ORDER (matched on stems, so either extension slots in)."""
    out = []
    for track in ("ml", "dsa"):
        for d in sorted((ROOT / track).iterdir()):
            if not d.is_dir():
                continue
            files = _notebook_files(d)
            if not files:
                continue
            order = [Path(n).stem for n in NB_ORDER.get(d.name, [])]
            files.sort(key=lambda p: (order.index(p.stem) if p.stem in order
                                      else len(order), p.name))
            notes = next((str((d / n).relative_to(ROOT))
                          for n in ("notes.md", "README.md")
                          if (d / n).is_file()), None)
            nbs = [{"path": str(p.relative_to(ROOT)), "file": p.name,
                    "title": _file_meta(p)[0], "blurb": _file_meta(p)[1]}
                   for p in files]
            out.append({"track": track, "folder": d.name,
                        "folder_title": _folder_title(track, d.name),
                        "notes_path": notes, "notebooks": nbs})
    return out


# ---------------------------------------------------------------------------
# Notes tree, markdown, flashcards
# ---------------------------------------------------------------------------
def _md_title(path):
    try:
        for line in path.read_text().splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return path.name


def _md_entry(path):
    return {"path": str(path.relative_to(ROOT)), "title": _md_title(path),
            "name": path.name}


def _reading_order(f):
    """Pedagogical order within a folder, not alphabetical: read the notes,
    study the implementations, then practice - solutions dead last so they
    never sit above the drill they spoil."""
    n = f.name.lower()
    if n == "readme.md":
        rank = 0
    elif n == "notes.md":
        rank = 1
    elif n == "problems.py":
        rank = 5  # full reference solutions: below the drills they'd spoil
    elif "solution" in n:
        rank = 5
    elif n in ("exercises.md", "drills.md"):
        rank = 4
    elif f.suffix == ".md":
        rank = 2  # topic notes read before the code they prepare for
    else:
        rank = 3  # .py / .ipynb after the reading
    return (rank, n)


def notes_tree():
    groups = []
    for track in ("ml", "dsa"):
        folders = []
        readme = ROOT / track / "README.md"
        if readme.exists():
            folders.append({"name": "overview", "title": "Overview",
                            "files": [_md_entry(readme)]})
        for d in sorted((ROOT / track).iterdir()):
            if d.is_dir():
                paths = sorted(list(d.glob("*.md")) + _notebook_files(d),
                               key=_reading_order)
                files = [_md_entry(f) if f.suffix == ".md" else
                         {"path": str(f.relative_to(ROOT)),
                          "title": ("Reference solutions"
                                    if f.name == "problems.py"
                                    else _file_meta(f)[0]),
                          "name": f.name} for f in paths]
                if files:
                    folders.append({"name": d.name,
                                    "title": _folder_title(track, d.name),
                                    "files": files})
        groups.append({"group": track, "folders": folders})
    pfiles = [_md_entry(f) for f in sorted((ROOT / "practice").glob("*.md"))]
    groups.append({"group": "practice",
                   "folders": [{"name": "practice", "title": "Practice system",
                                "files": pfiles}]})
    return groups


def read_md(raw_path):
    """Resolve and serve a markdown, python, or ipynb file - must stay inside
    the repo root. Python sources are wrapped in a fenced block so the client's
    existing markdown renderer shows them as read-only highlighted code;
    notebooks render as markdown with their code cells fenced."""
    p = (ROOT / raw_path).resolve()
    ok = (p.is_relative_to(ROOT) and p.suffix in (".md", ".py", ".ipynb")
          and len(p.relative_to(ROOT).parts) >= 2
          and p.relative_to(ROOT).parts[0] in ("ml", "dsa", "practice"))
    if not ok:
        return None, (400, "path must be a .md, .py or .ipynb file under "
                           "ml/, dsa/ or practice/")
    if not p.is_file():
        return None, (404, "no such file")
    if p.suffix == ".ipynb":
        # no header note: the breadcrumb already names the file and carries
        # the Open-as-notebook action; the first markdown cell is the title
        parts = []
        for kind, src in _ipynb_cells(p):
            src = src.rstrip("\n")
            if kind == "prose":
                parts.append(src)
            elif src:
                parts.append("```python\n" + src.replace("```", "`​``") + "\n```")
        return "\n\n".join(parts), None
    text = p.read_text()
    if p.name == "problems.py" and p.relative_to(ROOT).parts[0] == "dsa":
        return _problems_reference_md(p), None
    if p.suffix == ".py":
        rel = p.relative_to(ROOT)
        text = (f"# `{rel}`\n\n*Read-only view. To work through it cell by "
                f"cell, use **Open as notebook** above; to edit and run the "
                f"whole file, use **Playground** - or open `{rel}` in your "
                f"editor and run `uv run python {rel}`.*\n\n"
                "```python\n" + text.replace("```", "`​``") + "\n```")
    return text, None


def _problems_reference_md(path):
    """Render a dsa problems.py as a structured reference page instead of one
    giant code block: a section per problem with the statement visible and the
    hint and solution folded, so headings feed the on-page outline and nothing
    spoils on scroll."""
    data = _parse_problems_file(path)
    folder = path.parent.name
    rel = path.relative_to(ROOT)
    fence = lambda s: "```python\n" + s.replace("```", "`​``") + "\n```"
    parts = [
        f"# {FOLDER_TITLES.get(folder, folder)}: reference solutions",
        f"*`{rel}` - {len(data['problems'])} solved problems with "
        f"{data['assert_count']} asserts. Drill these from the **Problems** tab "
        f"(closed-book, timed); this page is for review after an attempt. "
        f"Hints and solutions are folded so scrolling spoils nothing.*",
    ]
    for pr in data["problems"]:
        parts.append(f"## {pr['name']}")
        parts.append(f"`{pr['difficulty']}` · {pr['kind']} · "
                     f"time box {pr['time_box_min']} min")
        parts.append(pr["statement"])
        if pr["helpers"]:
            parts.append("<details><summary>Provided helpers</summary>\n\n"
                         + fence(pr["helpers"]) + "\n</details>")
        if pr["key_insight"]:
            parts.append("<details><summary>Key insight</summary>\n\n"
                         + pr["key_insight"] + "\n</details>")
        parts.append("<details><summary>Solution</summary>\n\n"
                     + fence(pr["solution_source"]) + "\n</details>")
    return "\n\n".join(parts)


def _slug(text):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def parse_flashcards():
    """Parse practice/flashcards.md: '## section' headings, then repeated
    '**Q:** ...' lines followed by '<details><summary>A</summary> ... </details>'."""
    text = (ROOT / "practice" / "flashcards.md").read_text()
    cards, section, slug, per_section = [], "", "", 0
    question, answer, mode = [], [], None  # mode: None | 'q' | 'a'

    def flush():
        nonlocal question, answer, mode, per_section
        if question and answer:
            per_section += 1
            cards.append({"section": section, "slug": slug, "n": per_section,
                          "question": " ".join(question).strip(),
                          "answer": "\n".join(answer).strip()})
        question, answer, mode = [], [], None

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            slug = _slug(section)
            per_section = 0
        elif line.startswith("**Q:**"):
            flush()
            question = [line[6:].strip()]
            mode = "q"
        elif line.lstrip().startswith("<details"):
            if mode == "q":
                mode = "a"
        elif line.strip() == "</details>":
            flush()
        elif mode == "q" and line.strip() and not line.startswith("---"):
            question.append(line.strip())
        elif mode == "a":
            answer.append(line)
    flush()
    return cards


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s %s\n" % (self.command or "-", self.path))

    def _send(self, status, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj))

    def _err(self, status, msg):
        self._json({"error": msg}, status)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, (APP_DIR / "index.html").read_bytes(),
                           "text/html; charset=utf-8")
            elif u.path == "/api/problems":
                rows = queue_load()
                out = [{k: p[k] for k in ("id", "name", "kind", "folder",
                                          "folder_title", "statement_first_line",
                                          "difficulty", "time_box_min")}
                       | {"status": problem_status(p["id"], rows)}
                       for p in all_problems()]
                self._json(out)
            elif u.path == "/api/problem":
                pid = (qs.get("id") or [""])[0]
                p = find_problem(pid)
                if p is None:
                    return self._err(404, f"unknown problem id: {pid}")
                keys = ("id", "name", "kind", "folder", "folder_title",
                        "statement", "stub", "key_insight", "solution_source",
                        "helpers", "difficulty", "time_box_min", "source_path")
                self._json({k: p[k] for k in keys})
            elif u.path == "/api/tree":
                self._json(notes_tree())
            elif u.path == "/api/md":
                raw = (qs.get("path") or [""])[0]
                content, err = read_md(raw)
                if err:
                    return self._err(*err)
                self._json({"path": raw, "content": content})
            elif u.path == "/api/source":
                raw = (qs.get("path") or [""])[0]
                p, err = _lab_path(raw)
                if err:
                    return self._err(*err)
                self._json({"path": raw, "content": p.read_text()})
            elif u.path == "/api/nb/cells":
                raw = (qs.get("path") or [""])[0]
                result, status = nb_cells(raw)
                self._json(result, status)
            elif u.path == "/api/notebooks":
                self._json(notebooks_index())
            elif u.path == "/api/flashcards":
                self._json(parse_flashcards())
            elif u.path == "/api/queue":
                self._json(queue_snapshot())
            elif u.path == "/api/activity":
                self._json(activity())
            elif u.path.endswith((".md", ".py", ".ipynb")):
                # a repo file hit directly (hand-typed or stale link):
                # bounce into the SPA's notes view instead of a JSON 404
                rel = u.path.lstrip("/")
                if (ROOT / rel).is_file():
                    self.send_response(302)
                    self.send_header("Location", "/#/notes/" + rel)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self._err(404, "not found")
            else:
                self._err(404, "not found")
        except Exception as e:  # keep the local tool honest about failures
            self._err(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._err(400, "invalid Content-Length")
        if length > MAX_BODY:
            self.close_connection = True  # body is unread; don't reuse socket
            return self._err(413, f"request body too large (limit {MAX_BODY} bytes)")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._err(400, "invalid JSON body")
        try:
            if u.path == "/api/run":
                pid, code = body.get("id", ""), body.get("code", "")
                if not pid or not isinstance(code, str) or not code.strip():
                    return self._err(400, "need id and code")
                result, status = run_judge(pid, code)
                self._json(result, status)
            elif u.path == "/api/runfile":
                raw, code = body.get("path", ""), body.get("code", "")
                if not raw or not isinstance(code, str) or not code.strip():
                    return self._err(400, "need path and code")
                result, status = run_lab(raw, code)
                self._json(result, status)
            elif u.path == "/api/nb/start":
                raw = body.get("path", "")
                if not raw or not isinstance(raw, str):
                    return self._err(400, "need path")
                result, status = nb_start(raw)
                self._json(result, status)
            elif u.path == "/api/nb/run":
                sid, code = body.get("session", ""), body.get("code", "")
                if not sid or not isinstance(code, str):
                    return self._err(400, "need session and code")
                result, status = nb_run(sid, code)
                self._json(result, status)
            elif u.path == "/api/nb/restart":
                result, status = nb_restart(body.get("session", ""))
                self._json(result, status)
            elif u.path == "/api/nb/stop":
                result, status = nb_stop(body.get("session", ""))
                self._json(result, status)
            elif u.path == "/api/log":
                item, result = body.get("item_id", ""), body.get("result", "")
                if not item or result not in ("pass", "fail"):
                    return self._err(400, "need item_id and result pass|fail")
                self._json(queue_log(item, body.get("source_path", ""), result))
            else:
                self._err(404, "not found")
        except Exception as e:
            self._err(500, f"{type(e).__name__}: {e}")


def main():
    port = int(os.environ.get("PORT", "8888"))
    n = len(all_problems())  # warm the cache, fail fast on parse errors
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"MLPrep study app: http://127.0.0.1:{port}  ({n} problems discovered)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
