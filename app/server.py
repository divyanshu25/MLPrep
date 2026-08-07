#!/usr/bin/env python3
"""MLPrep study app server — stdlib only (Python 3.11).

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
    not the statement — showing it up front would spoil the problem.
    """
    paras = re.split(r"\n\s*\n", doc.strip())
    stmt, insight = [], []
    for p in paras:
        head = p.strip().lower()
        if insight or (stmt and (head.startswith("pattern") or
                                 head.startswith("key insight") or
                                 head.startswith("complexity"))):
            insight.append(p)
        else:
            stmt.append(p)
    return "\n\n".join(stmt), "\n\n".join(insight)


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
    """Signature + statement docstring + pass — the solution body is dropped."""
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
        if not statement:  # docstring opens with "Key insight" — keep 1st para
            statement, insight = doc.split("\n\n")[0], insight or doc
        problems.append({
            "id": f"{folder}/{node.name}",
            "name": node.name,
            "kind": "class" if isinstance(node, ast.ClassDef) else "function",
            "folder": folder,
            "folder_title": FOLDER_TITLES.get(folder, folder),
            "statement": statement,
            "statement_first_line": statement.strip().split("\n")[0],
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
        out.extend(_parse_problems_file(f)["problems"])
    return out


def find_problem(pid):
    for p in all_problems():
        if p["id"] == pid:
            return p
    return None


# ---------------------------------------------------------------------------
# Queue (mirrors practice/review.py's file format and mechanics exactly)
# ---------------------------------------------------------------------------
def queue_load():
    if not QUEUE.exists():
        QUEUE.write_text("\t".join(QUEUE_COLS) + "\n")
    lines = QUEUE.read_text().splitlines()[1:]
    return [dict(zip(QUEUE_COLS, ln.split("\t"))) for ln in lines if ln.strip()]

def queue_save(rows):
    out = ["\t".join(QUEUE_COLS)] + ["\t".join(r[c] for c in QUEUE_COLS) for r in rows]
    QUEUE.write_text("\n".join(out) + "\n")


def queue_log(item_id, source_path, result):
    """add-if-absent (interval 1), then pass doubles (cap 16) / fail resets."""
    with _queue_lock:
        rows = queue_load()
        row = next((r for r in rows if r["item_id"] == item_id), None)
        today = date.today()
        if row is None:
            row = {"item_id": item_id, "source_path": source_path or "",
                   "added": today.isoformat(), "last_result": "new",
                   "interval_days": "1",
                   "due": (today + timedelta(days=1)).isoformat()}
            rows.append(row)
        iv = min(int(row["interval_days"]) * 2, QUEUE_CAP) if result == "pass" else 1
        row.update(last_result=result, interval_days=str(iv),
                   due=(today + timedelta(days=iv)).isoformat())
        queue_save(rows)
        return row


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
    return {"rows": rows, "due_today": due, "today": today.isoformat(),
            "histogram": [{"interval": k, "count": v} for k, v in sorted(buckets.items())],
            "reviewed_today": reviewed_today}


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
    — only the target def is replaced by the user's code. ok:true therefore
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
            tgt = {t.id for t in n.targets if isinstance(t, ast.Name)}
            if tgt and not (tgt & existing):
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
                    "error": "Timed out after 30 s — likely an infinite loop.",
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


def notes_tree():
    groups = []
    for track in ("ml", "dsa"):
        folders = []
        readme = ROOT / track / "README.md"
        if readme.exists():
            folders.append({"name": "overview", "files": [_md_entry(readme)]})
        for d in sorted((ROOT / track).iterdir()):
            if d.is_dir():
                files = [_md_entry(f) for f in sorted(d.glob("*.md"))]
                if files:
                    folders.append({"name": d.name, "files": files})
        groups.append({"group": track, "folders": folders})
    pfiles = [_md_entry(f) for f in sorted((ROOT / "practice").glob("*.md"))]
    groups.append({"group": "practice",
                   "folders": [{"name": "practice", "files": pfiles}]})
    return groups


def read_md(raw_path):
    """Resolve and serve a markdown file — must stay inside the repo root."""
    p = (ROOT / raw_path).resolve()
    if not p.is_relative_to(ROOT) or p.suffix != ".md":
        return None, (400, "path must be a .md file inside the repo")
    if not p.is_file():
        return None, (404, "no such file")
    return p.read_text(), None


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
            elif u.path == "/api/flashcards":
                self._json(parse_flashcards())
            elif u.path == "/api/queue":
                self._json(queue_snapshot())
            else:
                self._err(404, "not found")
        except Exception as e:  # keep the local tool honest about failures
            self._err(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
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
            elif u.path == "/api/log":
                item, result = body.get("item_id", ""), body.get("result", "")
                if not item or result not in ("pass", "fail"):
                    return self._err(400, "need item_id and result pass|fail")
                row = queue_log(item, body.get("source_path", ""), result)
                self._json({"logged": True, "row": row})
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
