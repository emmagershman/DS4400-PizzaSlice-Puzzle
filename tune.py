"""Grid search over settings for the Slice Crawl estimators.

    python tune.py bayes                  # grid over (n0, threshold), seed 0
    python tune.py ci                     # grid over (n0, z), seed 0
    python tune.py bayes --confirm 3 0.99 # check one setting on seed 1
    python tune.py ci --confirm 3 2.5

Tunes on seed 0 (the "training" crawls); confirm the winner on seed 1
(fresh crawls) so we don't overfit our settings to one set of crawls.

Runs crawls in parallel across CPU cores. Uses the same crawl seeding as
slice_crawl.evaluate(), so results match evaluate() exactly.
"""
import csv
import sys
from multiprocessing import Pool

import slice_bayes
import slice_ci
from slice_crawl import SliceCrawl

# method -> (estimator builder, name of 2nd knob, n0 grid, 2nd-knob grid)
METHODS = {
    'bayes': (slice_bayes.make_estimator, 'threshold',
              [2, 3, 4, 6], [0.95, 0.97, 0.99, 0.995]),
    'ci': (slice_ci.make_estimator, 'z',
           [2, 3, 4, 6], [1.5, 2.0, 2.5, 3.0]),
}


def run_one(args):
    """Run one crawl; return (correct, n_slices)."""
    method, n0, knob, seed, idx = args
    build, knob_name = METHODS[method][:2]
    crawl = SliceCrawl.sample(seed=seed * 1_000_003 + idx)
    guess = build(n0=n0, **{knob_name: knob})(crawl)
    return guess == crawl._best, crawl.n_taste


def score(pool, method, n0, knob, seed, n_crawl):
    """Accuracy, its standard error, and mean slices over n_crawl crawls."""
    jobs = [(method, n0, knob, seed, i) for i in range(n_crawl)]
    out = pool.map(run_one, jobs, chunksize=10)
    acc = sum(c for c, _ in out) / n_crawl
    se = (acc * (1 - acc) / n_crawl) ** 0.5
    slices = sum(s for _, s in out) / n_crawl
    return acc, se, slices


def grid(method, n_crawl=1000, seed=0):
    _, knob_name, n0_grid, knob_grid = METHODS[method]
    path = f'tune_{method}.csv'
    with Pool() as pool, open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n0', knob_name, 'accuracy', 'accuracy_se',
                    'mean_slices'])
        for n0 in n0_grid:
            for knob in knob_grid:
                acc, se, slices = score(pool, method, n0, knob, seed, n_crawl)
                w.writerow([n0, knob, acc, se, slices])
                f.flush()
                print(f'{method} n0={n0} {knob_name}={knob}: acc {acc:.1%} '
                      f'(+/-{se:.1%}), {slices:.1f} slices', flush=True)
    print(f'saved {path}')


def confirm(method, n0, knob, n_crawl=2000, seed=1):
    knob_name = METHODS[method][1]
    with Pool() as pool:
        acc, se, slices = score(pool, method, n0, knob, seed, n_crawl)
    print(f'CONFIRM {method} (seed {seed}) n0={n0} {knob_name}={knob}: '
          f'acc {acc:.1%} (+/-{se:.1%}), {slices:.1f} slices')


if __name__ == '__main__':
    method = sys.argv[1]
    if len(sys.argv) == 5 and sys.argv[2] == '--confirm':
        confirm(method, int(sys.argv[3]), float(sys.argv[4]))
    else:
        grid(method)
