# Puzzle: Slice Crawl

There are 8 pizza shops near campus and you want to find the best one.
The catch: you learn about a shop only by tasting its slices.

**Your goal: taste as few slices as possible while still naming the best
shop with at least 95% confidence** (right in at least 95% of crawls).

Each slice gets a rating, but slices vary: the same shop can be great on
Tuesday and soggy on Friday. Each shop's ratings follow a bell curve around
that shop's typical rating, and some shops are more consistent than others.
You don't know any shop's typical rating or consistency.

`SliceCrawl` (in `slice_crawl.py`) runs one crawl. `crawl.taste(i)` buys a
slice from shop `i` and returns its rating. Write `estimate(crawl)`: taste
any shops you like, as many times as you like, then return the shop you
think is best.

- First, clear the bar: name the true best shop in at least 95% of crawls.
- Then, save pizza: among estimators that clear the bar, fewer slices
  (averaged over many crawls) is better.
- Practice on the
  [Slice Crawl site](https://matthigger.github.io/ml_puzzle/slice_crawl/#practice):
  it runs your `estimate()` on as many random crawls as you like and shows
  how it did.
- For fun, we'll race everyone's estimator head to head: fewest slices
  among those that clear 95% wins bragging rights (no effect on your grade).
  Post yours on the site's Leaderboard tab with class code `slice-n30b-xqvq-92e4`.

`evaluate(estimate)` runs your estimator on 200 random crawls and reports
how often it was right and how many slices it tasted on average (the same
`seed` always gives the same crawls). Anything on a crawl that starts with
an underscore (e.g. `crawl._mean`) is the hidden truth: peeking is
cheating.
