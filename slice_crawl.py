"""Slice Crawl: find the best pizza shop in as few slices as possible.

A crawl hides n_shop pizza shops. Each shop has its own typical rating (the
mean of its ratings) and its own consistency (their standard deviation).
Every slice you taste returns one noisy rating. Your estimator tastes as many
slices as it likes, then names the shop it believes has the highest typical
rating. It should be right in at least 95% of crawls while tasting as few
slices as it can.

    from slice_crawl import SliceCrawl, evaluate

    def estimate(crawl):
        for i in range(crawl.n_shop):
            crawl.taste(i)
        ...
        return best_shop_idx

    evaluate(estimate)          # 200 crawls of the official setup

Only the standard library is used, so the same file runs in a notebook and
in the browser (Pyodide) on the practice site.
"""
import math
import random

# the graded setup
OFFICIAL = dict(n_shop=8, family='bell', race='official')

# shapes of a shop's ratings: 'bell' is graded, the others are stress tests
FAMILY = {
    'bell': 'bell curve around each shop\'s typical rating',
    'heavy': 'mostly a bell curve, but now and then a wildly off slice',
    'disaster': 'usually fine, sometimes a disaster (burnt, cold, dropped)',
}

# (low, high) of the gap between the best and second-best typical rating.
# Log-uniform, so crawls range from blowouts to near-ties.
RACE = {
    'easy': (1.0, 3.0),
    'official': (0.3, 1.5),
    'close': (0.1, 0.4),
}

# far more than any sensible strategy needs: catches endless loops
MAX_TASTE = 20_000


class TooManySlices(RuntimeError):
    """Raise when a crawl has tasted MAX_TASTE slices."""


class SliceCrawl:
    """One crawl through n_shop pizza shops.

    Attributes you may use:
        n_shop (int): shops are numbered 0, 1, ..., n_shop - 1
        n_taste (int): slices tasted so far
        history (list): (shop_idx, rating) of every slice so far, in
            tasting order

    Everything starting with an underscore is the hidden truth: peeking at
    it is cheating (and the competition uses a fresh copy without it).
        _mean (list): (n_shop,) typical rating of each shop
        _sd (list): (n_shop,) standard deviation of each shop's ratings
    """

    def __init__(self, mean, sd, family='bell', seed=None):
        """Build a crawl with known shops (see sample() for random ones).

        Args:
            mean (list): (n_shop,) typical rating of each shop
            sd (list): (n_shop,) standard deviation of each shop
            family (str): shape of each shop's ratings, a key of FAMILY
            seed (int): seeds the slices
        """
        assert len(mean) == len(sd)
        self._mean = list(mean)
        self._sd = list(sd)
        self._family = family
        self._rng = random.Random(seed)
        self.n_shop = len(mean)
        self.n_taste = 0
        self.history = []

    @classmethod
    def sample(cls, n_shop=8, family='bell', race='official', seed=None):
        """Build a random crawl.

        Args:
            n_shop (int or tuple): number of shops, or (low, high) to draw
                it uniformly (inclusive) for each crawl
            family (str): shape of each shop's ratings, a key of FAMILY
            race (str): how close the best two shops are, a key of RACE
            seed (int): seeds both the shops and the slices

        Returns:
            crawl (SliceCrawl): a fresh crawl, nothing tasted yet
        """
        rng = random.Random(seed)
        if not isinstance(n_shop, int):
            n_shop = rng.randint(*n_shop)
        assert n_shop >= 2, 'need at least 2 shops'

        # Typical ratings: best, second a RACE gap below, the rest below
        # second, so the planted gap is the true margin of victory.
        lo, hi = RACE[race]
        best = rng.uniform(6.5, 8.5)
        second = best - math.exp(rng.uniform(math.log(lo), math.log(hi)))
        mean = [best, second] + [second - rng.uniform(0, 3)
                                 for _ in range(n_shop - 2)]

        # consistency: sd varies 5x across shops
        sd = [math.exp(rng.uniform(math.log(.5), math.log(2.5)))
              for _ in range(n_shop)]

        # shuffle so shop 0 isn't always best
        order = list(range(n_shop))
        rng.shuffle(order)
        mean = [mean[i] for i in order]
        sd = [sd[i] for i in order]

        # one seed fixes both the shops and every slice they serve
        return cls(mean, sd, family=family, seed=rng.getrandbits(32))

    def taste(self, shop_idx):
        """Buy one slice from shop shop_idx and return its rating.

        Raises:
            IndexError: shop_idx isn't a shop number
            TooManySlices: the crawl has already tasted MAX_TASTE slices
        """
        if not (isinstance(shop_idx, int) and 0 <= shop_idx < self.n_shop):
            raise IndexError(f'no shop {shop_idx!r}: shops are 0 to '
                             f'{self.n_shop - 1}')
        if self.n_taste >= MAX_TASTE:
            raise TooManySlices(f'stopped after {MAX_TASTE} slices')

        # every family has mean m and sd s
        m, s, rng = self._mean[shop_idx], self._sd[shop_idx], self._rng
        if self._family == 'bell':
            x = rng.gauss(m, s)
        elif self._family == 'heavy':
            # Student t with df = 3 (normal / sqrt(chi-squared / df)),
            # rescaled from variance df / (df - 2) to sd s
            df = 3
            t = rng.gauss(0, 1) / math.sqrt(rng.gammavariate(df / 2, 2) / df)
            x = m + s * t / math.sqrt(df / (df - 2))
        elif self._family == 'disaster':
            # 1 slice in 10 is 4 points low; + .4 keeps the mean at m
            x = rng.gauss(m, s)
            if rng.random() < .1:
                x -= 4
            x += .4
        else:
            raise ValueError(f'unknown family {self._family!r}')

        self.n_taste += 1
        self.history.append((shop_idx, x))
        return x

    @property
    def _best(self):
        """Return the true best shop (hidden truth, for scoring)."""
        return max(range(self.n_shop), key=self._mean.__getitem__)


def evaluate(estimate, n_crawl=200, seed=0, n_shop=8, family='bell',
             race='official', verbose=True):
    """Run estimate on n_crawl random crawls and summarize how it did.

    The same seed (and setup) always gives the same crawls. Tune on one
    seed and report on another. Accuracy is itself an estimate: over 200
    crawls it can be off by a few points by chance, so use thousands before
    trusting a pass.

    Args:
        estimate (function): tastes a crawl, returns the shop it names
        n_crawl (int): number of crawls
        seed (int): which set of crawls
        n_shop, family, race: setup, see SliceCrawl.sample()
        verbose (bool): print a one-line summary

    Returns:
        result (dict): {accuracy, accuracy_se, mean_slices, median_slices,
            p90_slices, slices, correct}. accuracy is the fraction of crawls
            named right, accuracy_se its standard deviation across repeat
            runs of n_crawl crawls; slices and correct are (n_crawl,)
            lists, one entry per crawl
    """
    slices, correct = [], []
    for idx in range(n_crawl):
        # prime stride: different seeds never share crawls
        crawl = SliceCrawl.sample(n_shop=n_shop, family=family, race=race,
                                  seed=seed * 1_000_003 + idx)
        guess = estimate(crawl)
        slices.append(crawl.n_taste)
        correct.append(guess == crawl._best)

    acc = sum(correct) / n_crawl
    srt = sorted(slices)
    result = dict(accuracy=acc,
                  accuracy_se=math.sqrt(acc * (1 - acc) / n_crawl),
                  mean_slices=sum(slices) / n_crawl,
                  median_slices=srt[n_crawl // 2],
                  p90_slices=srt[min(n_crawl - 1, int(.9 * n_crawl))],
                  slices=slices, correct=correct)
    if verbose:
        print(f'{n_crawl} crawls: right {acc:.1%} of the time '
              f'(need 95%), {result["mean_slices"]:.1f} slices on average')
    return result
