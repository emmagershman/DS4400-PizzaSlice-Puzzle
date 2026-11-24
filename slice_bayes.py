"""Bayesian estimator for Slice Crawl.

Idea: keep a posterior (belief distribution) over each shop's true mean,
estimate P(shop is best) by simulation, and stop once the leader's P(best)
clears a threshold. Standard library only, so it also runs on the practice
site.
"""
import math
import random


def summarize(ratings):
    """Return (mean, standard error, degrees of freedom) of one shop's ratings.

    The posterior for the shop's true mean is a Student-t centred on the
    sample mean with spread s / sqrt(n) and n - 1 degrees of freedom. The t
    has fatter tails than a normal, which guards against overconfidence when
    n is small and s is a shaky estimate.
    """
    n = len(ratings)
    mean = sum(ratings) / n
    s = math.sqrt(sum((x - mean) ** 2 for x in ratings) / (n - 1))
    return mean, s / math.sqrt(n), n - 1


def draw_t(df, rng):
    """One draw from a Student-t: normal / sqrt(chi-squared / df)."""
    return rng.gauss(0, 1) / math.sqrt(rng.gammavariate(df / 2, 2) / df)


def prob_best(posts, rng, n_draw):
    """Estimate P(shop is best) for every shop by Monte Carlo.

    Each round draws one plausible true mean per shop from its posterior and
    records which shop wins. A shop's P(best) is its fraction of wins.

    Args:
        posts (list): (mean, std_err, df) for each shop, from summarize()
    """
    wins = [0] * len(posts)
    for _ in range(n_draw):
        draws = [m + se * draw_t(df, rng) for m, se, df in posts]
        wins[draws.index(max(draws))] += 1
    return [w / n_draw for w in wins]


def sample_winner(posts, rng):
    """Draw one plausible mean per shop and return which shop wins."""
    draws = [m + se * draw_t(df, rng) for m, se, df in posts]
    return draws.index(max(draws))


def top_two_pick(posts, rng, max_tries=100):
    """Choose the next shop to taste (top-two Thompson sampling).

    Draw a winner A from the posteriors. Half the time taste A. Otherwise
    keep drawing until a different shop B wins, and taste B: the strongest
    challenger to A. Every shop gets tasted roughly in proportion to how
    plausibly it is the best (or the best challenger), so a shop that got
    unlucky early still gets a second chance.
    """
    a = sample_winner(posts, rng)
    if rng.random() < 0.5:
        return a
    for _ in range(max_tries):
        b = sample_winner(posts, rng)
        if b != a:
            return b
    return a


def make_estimator(n0=3, threshold=0.99, n_draw=500, seed=0):
    """Build an estimate(crawl) function with the given settings.

    Args:
        n0 (int): warm-up slices per shop (at least 2, to compute s)
        threshold (float): stop when the leader's P(best) reaches this
        n_draw (int): Monte Carlo draws per P(best) estimate
        seed (int): seeds the posterior draws, so runs are repeatable
    """
    assert n0 >= 2, 'need 2 slices per shop to estimate its sd'

    def estimate(crawl):
        rng = random.Random(seed)
        ratings = [[crawl.taste(i) for _ in range(n0)]
                   for i in range(crawl.n_shop)]
        posts = [summarize(r) for r in ratings]

        while True:
            p = prob_best(posts, rng, n_draw)
            leader = p.index(max(p))
            if p[leader] >= threshold:
                return leader

            # P(best) is the slow part, so re-check it only every ~5% more
            # slices. This overshoots the stopping point by at most ~5%.
            for _ in range(max(1, crawl.n_taste // 20)):
                pick = top_two_pick(posts, rng)
                ratings[pick].append(crawl.taste(pick))
                posts[pick] = summarize(ratings[pick])

    return estimate
