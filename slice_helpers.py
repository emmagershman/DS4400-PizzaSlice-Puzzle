"""Settings and helper functions for the estimator in slice_stud_higger.ipynb."""
import math

# ---- settings ----
N0 = 3             # step 1: slices per shop to start
THRESHOLD = 0.86 # step 5: stop when the leader's P(best) reaches this
N_ROUND = 500      # step 4: rounds of guesses per P(best) check
SEED = 0           # seeds the guesses, so reruns give identical results


def summarize(ratings):
    """Steps 2-3: (mean, standard error, df) of one shop's ratings."""
    n = len(ratings)
    mean = sum(ratings) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in ratings) / (n - 1))
    return mean, sd / math.sqrt(n), n - 1


def guess_winner(summaries, rng):
    """Step 4, one round: guess = mean + se * draw_t for every shop;
    return the shop with the highest guess."""
    guesses = []
    for mean, se, df in summaries:
        draw_t = rng.gauss(0, 1) / math.sqrt(rng.gammavariate(df / 2, 2) / df)
        guesses.append(mean + se * draw_t)
    return guesses.index(max(guesses))


def prob_best(summaries, rng):
    """Step 5: P(best) = wins / N_ROUND for every shop."""
    wins = [0] * len(summaries)
    for _ in range(N_ROUND):
        wins[guess_winner(summaries, rng)] += 1
    return [w / N_ROUND for w in wins]


def top_two_pick(summaries, rng):
    """Next shop to taste: guess a winner A; half the time taste A, otherwise
    keep guessing until a different shop B wins and taste B."""
    a = guess_winner(summaries, rng)
    if rng.random() < 0.5:
        return a
    for _ in range(100):
        b = guess_winner(summaries, rng)
        if b != a:
            return b
    return a


def taste_more(crawl, ratings, summaries, rng):
    """Step 6: taste ~5% more slices (at least 1), each chosen by
    top_two_pick, updating that shop's ratings and summary as we go.
    Re-checking P(best) only every ~5% keeps the estimator fast."""
    for _ in range(max(1, crawl.n_taste // 20)):
        pick = top_two_pick(summaries, rng)
        ratings[pick].append(crawl.taste(pick))
        summaries[pick] = summarize(ratings[pick])


# ---- experiments & plots (used only by the notebook's visualization cells) ----
# matplotlib / multiprocessing are imported inside these functions so the
# estimator code above still runs anywhere (e.g. the practice site).

GRID_N0 = [3, 4, 6]
GRID_THRESHOLD = [0.86, 0.88, 0.90, 0.92]
GRID_SEEDS = range(10, 35)     # 25 runs of 200 crawls, like 25 evaluate() calls
GRID_CSV = 'grid_results.csv'

INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID_LINE, SURFACE = '#e1e0d9', '#fcfcfb'
GOOD_WASH, FAIL_WASH = '#dcf2dc', '#f0efec'
SHOP_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
               '#e87ba4', '#008300', '#6250d6', '#e34948']


def run_crawl(crawl, n0=N0, threshold=THRESHOLD, trace=None):
    """The estimator with adjustable settings. If trace is a list, append
    (slices tasted, P(best) list) at every check, for plotting."""
    import random
    rng = random.Random(SEED)
    ratings = [[crawl.taste(i) for _ in range(n0)] for i in range(crawl.n_shop)]
    summaries = [summarize(r) for r in ratings]
    while True:
        p = prob_best(summaries, rng)
        if trace is not None:
            trace.append((crawl.n_taste, p))
        if max(p) >= threshold:
            return p.index(max(p))
        taste_more(crawl, ratings, summaries, rng)


def _grid_job(args):
    """One crawl of the grid: (n0, threshold, seed, correct, slices)."""
    from slice_crawl import SliceCrawl
    n0, thr, seed, idx = args
    crawl = SliceCrawl.sample(seed=seed * 1_000_003 + idx)   # same as evaluate()
    guess = run_crawl(crawl, n0, thr)
    return n0, thr, seed, guess == crawl._best, crawl.n_taste


def run_grid(path=GRID_CSV):
    """Run every (n0, threshold) on 25 runs of 200 crawls; save each run's
    accuracy and mean slices to a CSV. Slow (~20 min), so run it once."""
    import csv
    from multiprocessing import Pool
    jobs = [(n0, thr, s, i) for n0 in GRID_N0 for thr in GRID_THRESHOLD
            for s in GRID_SEEDS for i in range(200)]
    with Pool() as pool:
        out = pool.map(_grid_job, jobs, chunksize=20)
    runs = {}
    for n0, thr, seed, correct, slices in out:
        runs.setdefault((n0, thr, seed), []).append((correct, slices))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n0', 'threshold', 'seed', 'accuracy', 'mean_slices'])
        for (n0, thr, seed), r in runs.items():
            w.writerow([n0, thr, seed, sum(c for c, _ in r) / len(r),
                        sum(s for _, s in r) / len(r)])


def plot_grid_table(path=GRID_CSV, chosen=(N0, THRESHOLD)):
    """Table of (n0, threshold): accuracy over all crawls in the grid
    (25 runs x 200) and mean slices. Green = at least 95% correct.
    The chosen setting is outlined."""
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cells = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (int(row['n0']), float(row['threshold']))
            cells.setdefault(key, []).append(
                (float(row['accuracy']), float(row['mean_slices'])))

    # every run has 200 crawls, so averaging the runs = accuracy over all crawls
    acc = {k: sum(a for a, _ in v) / len(v) for k, v in cells.items()}
    sl = {k: sum(s for _, s in v) / len(v) for k, v in cells.items()}
    cheapest = min((k for k in cells if acc[k] >= 0.95), key=sl.get)

    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for r, n0 in enumerate(GRID_N0):
        for c, thr in enumerate(GRID_THRESHOLD):
            ok = acc[(n0, thr)] >= 0.95
            slices = sl[(n0, thr)]
            ax.add_patch(Rectangle((c + .03, r + .03), .94, .94, linewidth=0,
                                   facecolor=GOOD_WASH if ok else FAIL_WASH))
            mark = '✓ ' if ok else ''
            ax.text(c + .5, r + .4, f'{mark}{acc[(n0, thr)]:.1%} correct',
                    ha='center', va='center', fontsize=11, color=INK,
                    fontweight='bold' if ok else 'normal')
            ax.text(c + .5, r + .66, f'{slices:.0f} slices', ha='center',
                    va='center', fontsize=10, color=INK_2)
            if (n0, thr) == tuple(chosen):
                ax.add_patch(Rectangle((c + .03, r + .03), .94, .94,
                                       fill=False, edgecolor=INK, linewidth=2))

    ax.set_xlim(0, len(GRID_THRESHOLD))
    ax.set_ylim(len(GRID_N0), 0)
    ax.set_xticks([c + .5 for c in range(len(GRID_THRESHOLD))],
                  [f'{t:.2f}' for t in GRID_THRESHOLD])
    ax.set_yticks([r + .5 for r in range(len(GRID_N0))],
                  [str(n) for n in GRID_N0])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('P(best) threshold', color=INK_2)
    ax.set_ylabel('warm-up slices per shop (n0)', color=INK_2)
    ax.tick_params(length=0, colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    n_crawls = 200 * len(next(iter(cells.values())))
    fig.suptitle(f'Accuracy and average slices per crawl, over {n_crawls:,} '
                 'crawls per setting', color=INK, fontsize=12,
                 x=0.02, ha='left')
    why = (': the cheapest setting that clears 95%'
           if cheapest == tuple(chosen) else '')
    fig.text(0.02, 0.01, f'Green ✓ = at least 95% correct. '
             f'Outlined = our choice{why}.', color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()


def find_close_race(max_gap=0.5, slices=(120, 400), start=0):
    """Return the seed of a crawl where the top two shops are close, so the
    P(best) lines cross. Uses the hidden truth, but only to pick an example
    to plot -- the estimator itself never looks at it."""
    from slice_crawl import SliceCrawl
    seed = start
    while True:
        crawl = SliceCrawl.sample(seed=seed)
        top2 = sorted(crawl._mean)[-2:]
        if top2[1] - top2[0] < max_gap:
            run_crawl(crawl)
            if slices[0] <= crawl.n_taste <= slices[1]:
                return seed
        seed += 1


def plot_pbest(seed):
    """One crawl: P(best) of every shop vs slices tasted so far."""
    import matplotlib.pyplot as plt
    from slice_crawl import SliceCrawl

    crawl = SliceCrawl.sample(seed=seed)
    trace = []
    guess = run_crawl(crawl, trace=trace)
    x = [n for n, _ in trace]

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for shop in range(crawl.n_shop):
        y = [p[shop] for _, p in trace]
        ax.plot(x, y, color=SHOP_COLORS[shop], linewidth=2,
                drawstyle='steps-post', label=f'shop {shop}')
        if y[-1] > 0.05:                  # direct-label the final contenders
            ax.annotate(f'shop {shop}', (x[-1], y[-1]), xytext=(8, 0),
                        textcoords='offset points', va='center',
                        fontsize=9, color=INK_2)
    ax.plot(x[-1], trace[-1][1][guess], 'o', markersize=8, zorder=3,
            color=SHOP_COLORS[guess], markeredgecolor=SURFACE,
            markeredgewidth=2)                     # where we stopped
    ax.axhline(THRESHOLD, color=MUTED, linestyle='--', linewidth=1)
    ax.text(x[0], THRESHOLD + .02, f'stop at P(best) ≥ {THRESHOLD:.2f}',
            fontsize=9, color=MUTED)

    ax.set_xlabel('slices tasted so far', color=INK_2)
    ax.set_ylabel('P(best)', color=INK_2)
    ax.set_ylim(0, 1.02)
    ax.grid(axis='y', color=GRID_LINE, linewidth=.8)
    ax.tick_params(colors=INK_2, length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.legend(ncol=4, frameon=False, fontsize=9, loc='upper center',
              bbox_to_anchor=(.5, -.18), labelcolor=INK_2)
    verdict = 'right' if guess == crawl._best else 'wrong'
    ax.set_title(f'P(best) for each shop during one crawl (seed {seed}): '
                 f'named shop {guess} after {crawl.n_taste} slices ({verdict})',
                 color=INK, fontsize=11, loc='left')
    fig.tight_layout()
    plt.show()
