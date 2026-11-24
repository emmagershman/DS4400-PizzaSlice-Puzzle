"""Confidence-interval (successive elimination) estimator for Slice Crawl.

Idea: give every shop a confidence interval mean +/- z * s / sqrt(n) for its
true mean. A shop whose interval sits entirely below some other shop's
interval is (confidently) not the best, so we eliminate it and stop tasting
it. Keep tasting the survivors until only one is left.
Standard library only, so it also runs on the practice site.
"""
import math


def interval(ratings, z):
    """Return the (low, high) confidence interval for a shop's true mean."""
    n = len(ratings)
    mean = sum(ratings) / n
    s = math.sqrt(sum((x - mean) ** 2 for x in ratings) / (n - 1))
    half = z * s / math.sqrt(n)
    return mean - half, mean + half


def make_estimator(n0=3, z=3.0, max_slices=1000):
    """Build an estimate(crawl) function with the given settings.

    Args:
        n0 (int): warm-up slices per shop (at least 2, to compute s)
        z (float): interval half-width in standard errors. Bigger z means
            wider intervals: slower to eliminate, but fewer mistakes.
        max_slices (int): budget. If two survivors are nearly tied (e.g. the
            2nd and 3rd best, after the true best was wrongly eliminated)
            their intervals may never separate; at the budget we stop and
            return the survivor with the highest mean.
    """
    assert n0 >= 2, 'need 2 slices per shop to estimate its sd'

    def estimate(crawl):
        ratings = {i: [crawl.taste(i) for _ in range(n0)]
                   for i in range(crawl.n_shop)}
        ci = {i: interval(r, z) for i, r in ratings.items()}

        while len(ratings) > 1:
            # Eliminate every shop whose best case (upper bound) is below
            # the best worst case (highest lower bound) among survivors.
            best_low = max(low for low, _ in ci.values())
            for i in [i for i, (_, high) in ci.items() if high < best_low]:
                del ratings[i], ci[i]
            if len(ratings) == 1:
                break
            if crawl.n_taste >= max_slices:
                return max(ratings, key=lambda i: sum(ratings[i])
                           / len(ratings[i]))

            # Taste next: the survivor with the widest interval, i.e. the
            # one we are least sure about.
            pick = max(ci, key=lambda i: ci[i][1] - ci[i][0])
            ratings[pick].append(crawl.taste(pick))
            ci[pick] = interval(ratings[pick], z)

        return next(iter(ratings))

    return estimate
