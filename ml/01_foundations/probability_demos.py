"""Probability foundations, demonstrated numerically (numpy only, no plots).

Four demos, each a claim you can state on a whiteboard and then CHECK by
simulation:

  1. Law of large numbers: the sample mean converges to E[X].
  2. Central limit theorem: sample means become Gaussian (std shrinks like
     sigma/sqrt(n), skewness -> 0) even for lopsided source distributions.
  3. MLE for a coin: the p maximizing the log-likelihood is exactly the
     empirical frequency h/n (checked by brute-force argmax over a grid).
  4. Bayes rule: the medical-test example computed with raw counts, so the
     'surprising' posterior is just arithmetic on a population table.

Run:  python probability_demos.py
"""

import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# Demo 1: law of large numbers
# ---------------------------------------------------------------------------

def demo_lln():
    print("=" * 70)
    print("DEMO 1: Law of large numbers")
    print("=" * 70)
    print("Rolling a fair die: E[X] = (1+2+...+6)/6 = 3.5 exactly.")
    print("The running mean of n rolls should approach 3.5, with error")
    print("shrinking roughly like 1/sqrt(n).\n")

    true_mean = 3.5
    rolls = np.random.randint(1, 7, size=1_000_000)
    print(f"  {'n':>9s} {'sample mean':>12s} {'|error|':>10s} "
          f"{'~1/sqrt(n) scale':>17s}")
    for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        m = rolls[:n].mean()
        # std of one die roll is sqrt(35/12) ~= 1.708
        predicted_scale = np.sqrt(35 / 12) / np.sqrt(n)
        print(f"  {n:>9,d} {m:>12.4f} {abs(m - true_mean):>10.4f} "
              f"{predicted_scale:>17.4f}")
    print("\n  The observed error hovers around the predicted sigma/sqrt(n)")
    print("  scale: averaging works, and 4x the data buys half the error.\n")


# ---------------------------------------------------------------------------
# Demo 2: central limit theorem
# ---------------------------------------------------------------------------

def skewness(x):
    """Third standardized moment: 0 for any symmetric distribution."""
    x = np.asarray(x, dtype=float)
    m, s = x.mean(), x.std()
    return np.mean(((x - m) / s) ** 3)


def demo_clt():
    print("=" * 70)
    print("DEMO 2: Central limit theorem")
    print("=" * 70)
    print("Take means of n draws; repeat 20,000 times to see the mean's")
    print("distribution. CLT claims: std of the mean = sigma/sqrt(n), and the")
    print("shape becomes Gaussian (skewness -> 0) NO MATTER the source shape.\n")

    n_trials = 20_000
    sources = {
        # name: (sampler, true mean, true std, true skewness of ONE draw)
        "Uniform(0,1)": (lambda size: np.random.rand(*size),
                         0.5, np.sqrt(1 / 12), 0.0),
        "Exponential(1)": (lambda size: np.random.exponential(1.0, size=size),
                           1.0, 1.0, 2.0),
    }

    for name, (sampler, mu, sigma, skew1) in sources.items():
        print(f"  Source: {name}  (mean={mu}, std={sigma:.4f}, "
              f"skewness of one draw={skew1:.1f})")
        print(f"    {'n':>5s} {'std of means':>13s} {'sigma/sqrt(n)':>14s} "
              f"{'skew of means':>14s}")
        for n in [1, 5, 30, 100, 1000]:
            means = sampler((n_trials, n)).mean(axis=1)
            print(f"    {n:>5d} {means.std():>13.5f} {sigma / np.sqrt(n):>14.5f} "
                  f"{skewness(means):>14.4f}")
        print()

    print("  Read the table: measured std of the means tracks sigma/sqrt(n)")
    print("  almost perfectly, and even the very skewed exponential (skew 2.0)")
    print("  has its skewness crushed toward 0 as n grows -- the bell shape")
    print("  emerges from averaging, regardless of the source distribution.")
    print("  (Theory: skewness of a mean of n i.i.d. draws is skew1/sqrt(n),")
    print(f"  e.g. exponential at n=100 predicts {2.0/np.sqrt(100):.2f}.)\n")


# ---------------------------------------------------------------------------
# Demo 3: MLE for a coin flip
# ---------------------------------------------------------------------------

def demo_mle_coin():
    print("=" * 70)
    print("DEMO 3: MLE for a coin -- argmax of log-likelihood = h/n")
    print("=" * 70)
    true_p = 0.7
    n = 200
    flips = (np.random.rand(n) < true_p).astype(int)
    h = flips.sum()
    print(f"  Flip a p={true_p} coin {n} times: observed {h} heads.")
    print(f"  Empirical frequency h/n = {h / n:.4f}\n")

    print("  Log-likelihood of the data as a function of the parameter p:")
    print("    log L(p) = h*log(p) + (n-h)*log(1-p)")
    grid = np.linspace(0.001, 0.999, 999)
    log_lik = h * np.log(grid) + (n - h) * np.log(1 - grid)

    print(f"\n    {'p':>6s} {'log L(p)':>12s}")
    for p in [0.3, 0.5, 0.6, h / n, 0.8, 0.9]:
        ll = h * np.log(p) + (n - h) * np.log(1 - p)
        marker = "   <-- h/n" if abs(p - h / n) < 1e-12 else ""
        print(f"    {p:>6.3f} {ll:>12.4f}{marker}")

    p_hat = grid[np.argmax(log_lik)]
    print(f"\n  Brute-force argmax over a grid of 999 p values: "
          f"p_hat = {p_hat:.4f}")
    print(f"  Empirical frequency:                            "
          f"h/n   = {h / n:.4f}")
    assert abs(p_hat - h / n) < 1e-3, "MLE should equal empirical frequency"
    print("  They agree (to grid resolution), matching the calculus result:")
    print("  d/dp [h log p + (n-h) log(1-p)] = h/p - (n-h)/(1-p) = 0 "
          "=> p = h/n.")
    print("  MLE formalizes the obvious estimator -- and the SAME machinery")
    print("  (minimize negative log-likelihood) is exactly the cross-entropy")
    print("  loss used to train classifiers.\n")


# ---------------------------------------------------------------------------
# Demo 4: Bayes rule with raw counts
# ---------------------------------------------------------------------------

def demo_bayes():
    print("=" * 70)
    print("DEMO 4: Bayes rule -- the medical test, with raw counts")
    print("=" * 70)
    prevalence = 0.01     # P(sick)          -- the prior
    sensitivity = 0.99    # P(+ | sick)      -- true positive rate
    specificity = 0.95    # P(- | healthy)   -- true negative rate
    population = 100_000

    sick = int(population * prevalence)
    healthy = population - sick
    true_pos = int(sick * sensitivity)
    false_neg = sick - true_pos
    false_pos = int(healthy * (1 - specificity))
    true_neg = healthy - false_pos

    print(f"  Disease prevalence 1%, test sensitivity 99%, specificity 95%.")
    print(f"  Imagine a population of {population:,} people:\n")
    print(f"    {'':16s}{'test +':>10s}{'test -':>10s}{'total':>10s}")
    print(f"    {'sick':16s}{true_pos:>10,d}{false_neg:>10,d}{sick:>10,d}")
    print(f"    {'healthy':16s}{false_pos:>10,d}{true_neg:>10,d}{healthy:>10,d}")
    print(f"    {'total':16s}{true_pos + false_pos:>10,d}"
          f"{false_neg + true_neg:>10,d}{population:>10,d}\n")

    posterior_counts = true_pos / (true_pos + false_pos)
    print(f"  P(sick | +) from counts = {true_pos:,} / "
          f"({true_pos:,} + {false_pos:,}) = {posterior_counts:.4f}")

    # Same thing via the formula, to show it is the same arithmetic:
    p_pos = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    posterior_formula = sensitivity * prevalence / p_pos
    print(f"  P(sick | +) via Bayes   = P(+|sick)P(sick)/P(+) "
          f"= {sensitivity}*{prevalence}/{p_pos:.4f} = {posterior_formula:.4f}\n")

    print(f"  A 99%-sensitive test, yet a positive result means only a "
          f"~{posterior_formula:.0%} chance")
    print("  of disease. Why: false positives are drawn from the HUGE healthy")
    print(f"  pool ({false_pos:,}) and swamp the true positives ({true_pos:,}).")
    print("  The prior matters. Same math as precision under class imbalance:")
    print("  this is why 'the model is 99% accurate' is meaningless for rare")
    print("  events. Also check the simulation agrees:")

    # Monte Carlo confirmation
    n_sim = 1_000_000
    is_sick = np.random.rand(n_sim) < prevalence
    tests_pos = np.where(is_sick,
                         np.random.rand(n_sim) < sensitivity,
                         np.random.rand(n_sim) < (1 - specificity))
    mc = is_sick[tests_pos].mean()
    print(f"    Monte Carlo with {n_sim:,} simulated patients: "
          f"P(sick | +) = {mc:.4f}\n")


if __name__ == "__main__":
    demo_lln()
    demo_clt()
    demo_mle_coin()
    demo_bayes()
    print("All probability demos completed.")
