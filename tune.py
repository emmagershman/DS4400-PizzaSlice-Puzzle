"""Grid search over settings for the Slice Crawl estimators.

    python tune.py bayes                      # full grid, seed 0
    python tune.py ci
    python tune.py ci --confirm n0=6 z=1.25   # one setting on seed 1

Tunes on seed 0 (the "training" crawls); confirm the winner on seed 1
(fresh crawls) so we don't overfit our settings to one set of crawls.

Runs crawls in parallel across CPU cores. Uses the same crawl seeding as
slice_crawl.evaluate(), so results match evaluate() exactly.
"""
import csv
import itertools
import sys
from multiprocessing import Pool

import slice_bayes
import slice_ci
from slice_crawl import SliceCrawl

# method -> (estimator builder, {setting name: values to try})
METHODS = {
    'bayes': (slice_bayes.make_estimator,
              {'n0': [2, 3, 4, 6],
               'threshold': [0.95, 0.97, 0.99, 0.995]}),
    'ci': (slice_ci.make_estimator,
           {'n0': [2, 3, 4, 6], 'z': [1.5, 2.0, 2.5, 3.0]}),
}


def run_one(args):
    """Run one crawl; return (correct, n_slices)."""
    method, params, seed, idx = args
    crawl = SliceCrawl.sample(seed=seed * 1_000_003 + idx)
    guess = METHODS[method][0](**params)(crawl)
    return guess == crawl._best, crawl.n_taste


def score(pool, method, params, seed=0, n_crawl=1000):
    """Accuracy, its standard error, and mean slices over n_crawl crawls.

    Args:
        params (dict): settings for the method's make_estimator()
    """
    jobs = [(method, params, seed, i) for i in range(n_crawl)]
    out = pool.map(run_one, jobs, chunksize=10)
    acc = sum(c for c, _ in out) / n_crawl
    se = (acc * (1 - acc) / n_crawl) ** 0.5
    slices = sum(s for _, s in out) / n_crawl
    return acc, se, slices


def fmt(method, params, acc, se, slices):
    p = ' '.join(f'{k}={v}' for k, v in params.items())
    return f'{method} {p}: acc {acc:.1%} (+/-{se:.1%}), {slices:.1f} slices'


def grid(method, n_crawl=1000, seed=0):
    space = METHODS[method][1]
    names = list(space)
    path = f'tune_{method}.csv'
    with Pool() as pool, open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(names + ['accuracy', 'accuracy_se', 'mean_slices'])
        for values in itertools.product(*space.values()):
            params = dict(zip(names, values))
            acc, se, slices = score(pool, method, params, seed, n_crawl)
            w.writerow(list(values) + [acc, se, slices])
            f.flush()
            print(fmt(method, params, acc, se, slices), flush=True)
    print(f'saved {path}')


def confirm(method, params, n_crawl=2000, seed=1):
    with Pool() as pool:
        acc, se, slices = score(pool, method, params, seed, n_crawl)
    print(f'CONFIRM (seed {seed}) ' + fmt(method, params, acc, se, slices))


def parse(pairs):
    """['n0=6', 'z=1.25'] -> {'n0': 6, 'z': 1.25}"""
    out = {}
    for pair in pairs:
        k, v = pair.split('=')
        out[k] = int(v) if k == 'n0' else float(v)
    return out


if __name__ == '__main__':
    method = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == '--confirm':
        confirm(method, parse(sys.argv[3:]))
    else:
        grid(method)
