"""Settings and helper functions for the estimator in slice_stud_higger.ipynb."""
import math

# ---- settings ----
N0 = 3             # step 1: slices per shop to start
THRESHOLD = 0.86   # step 5: stop when the leader's P(best) reaches this
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
