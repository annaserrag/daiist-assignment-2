"""
Cost model for the seal check, shared by train.py and app.py so the dashboard and the
pipeline can never disagree about what a threshold costs.

The decision (see REPORT.md): during buy-side due diligence, every execution page in
the target's data room must carry the company seal. The detector looks at one page and
the page lands in one of three zones, using its stamp score (the confidence of the most
confident stamp box on the page, 0 if there is none):

    score >= high          pass     the page is treated as sealed, nobody looks at it
    low <= score < high    review   an associate checks the page by eye
    score < low            fail     the page is flagged as unsealed and an executed
                                    copy is requested from the seller's counsel

Positive class: an UNSEALED page (the thing that needs action), as with "defective" in
the Session 11 bottles. So an FN is an unsealed page we passed as sealed, and an FP is a
sealed page we flagged.

Every euro figure below is an assumption argued in REPORT.md, not a measurement.
"""

import numpy as np

STAMP_CLASS = "stamp"

# Costs per page, in EUR (REPORT.md, "Cost matrix and operating point").
COST_FN = 2_000.0       # unsealed page passed as sealed: 2% chance it becomes a
                        # EUR 100k problem after closing (indemnity, price chip)
COST_FLAG = 60.0        # any page flagged as unsealed (TP or FP): ten minutes of
                        # associate time to raise it with the seller, at EUR 375/hour
COST_TN = 0.0           # a sealed page passed: nothing happens
ASSOCIATE_RATE = 375.0  # EUR per hour, the A1 day rate (EUR 3,000 / 8 hours)
REVIEW_SECONDS = 30     # one page checked by eye for a seal
COST_REVIEW = ASSOCIATE_RATE * REVIEW_SECONDS / 3600  # about EUR 3.13 a page

# The associate is a second classifier, not an oracle (Session 11, section 6).
ASSOCIATE_RECALL = 0.95       # shown an unsealed page, notices 95% of the time
ASSOCIATE_SPECIFICITY = 0.99  # wrongly flags 1 sealed page in 100

BASE_RATE = 0.05        # share of execution pages in a data room with no seal
PAGES_PER_DAY = 600     # two live due diligences, ~3,000 execution pages each,
                        # reviewed over ten working days
REVIEW_HOURS_PER_DAY = 6  # productive hours one associate can give to the queue

PER = 1000

# One grid of cuts, reused by every sweep. Stamp scores bunch close to 1, so the top
# end gets a finer grid, the lesson from the Session 11 sweep.
CUTS = np.unique(np.round(np.concatenate([
    np.arange(0.0, 0.9, 0.01),
    np.arange(0.9, 1.0, 0.0025),
    [1.0, 1.01],
]), 4))


def page_score(boxes, names):
    """Confidence of the most confident stamp box, or 0.0 if the page has none.

    The maximum, not the mean: one confident seal is enough to call a page sealed, and
    averaging it against weaker boxes elsewhere would dilute exactly that evidence.
    """
    scores = [float(conf) for cls, conf in boxes if names[int(cls)] == STAMP_CLASS]
    return max(scores, default=0.0)


def zone_rates(scores, low, high):
    """How often pages of one truth class land in each zone: (pass, review, fail)."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(scores >= high)),
        float(np.mean((scores >= low) & (scores < high))),
        float(np.mean(scores < low)),
    )


def three_way_cost(
    unsealed_scores,
    sealed_scores,
    low,
    high,
    base_rate=BASE_RATE,
    cost_fn=COST_FN,
    cost_flag=COST_FLAG,
    cost_review=COST_REVIEW,
    associate_recall=ASSOCIATE_RECALL,
    associate_specificity=ASSOCIATE_SPECIFICITY,
):
    """Expected outcome per 1,000 pages at the real base rate.

    Zone rates are measured per truth class, so they belong to the model and survive
    the move from the test mix to the data room mix; only the base rate changes.
    A single cut is low == high (nothing reviewed).
    """
    pos_pass, pos_review, pos_fail = zone_rates(unsealed_scores, low, high)
    neg_pass, neg_review, neg_fail = zone_rates(sealed_scores, low, high)
    pos_n, neg_n = PER * base_rate, PER * (1 - base_rate)

    # An FN is an unsealed page passed by the model, or one the associate missed.
    fn = pos_n * (pos_pass + pos_review * (1 - associate_recall))
    tp = pos_n * (pos_fail + pos_review * associate_recall)
    fp = neg_n * (neg_fail + neg_review * (1 - associate_specificity))
    tn = neg_n - fp
    reviews = pos_n * pos_review + neg_n * neg_review
    flagged = tp + fp
    return {
        "low": float(low),
        "high": float(high),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "reviews": reviews,
        "review_share": reviews / PER,
        "flagged": flagged,
        "cost": fn * cost_fn + flagged * cost_flag + reviews * cost_review + tn * COST_TN,
    }


def single_cut_sweep(unsealed_scores, sealed_scores, **costs):
    """Cost per 1,000 pages for every single cut on the grid (no review band)."""
    return [three_way_cost(unsealed_scores, sealed_scores, c, c, **costs) for c in CUTS]


def rank_bands(unsealed_scores, sealed_scores, **costs):
    """Every (low, high) band on the grid, cheapest first."""
    options = [
        three_way_cost(unsealed_scores, sealed_scores, low, high, **costs)
        for i, low in enumerate(CUTS)
        for high in CUTS[i:]
    ]
    return sorted(options, key=lambda option: option["cost"])


def reference_designs(unsealed_scores, sealed_scores, **costs):
    """The three alternatives the brief asks the design to be priced against."""
    best_cut = min(single_cut_sweep(unsealed_scores, sealed_scores, **costs),
                   key=lambda row: row["cost"])
    return {
        # Pass everything: the seal is never checked.
        "do nothing": three_way_cost(unsealed_scores, sealed_scores, 0.0, 0.0, **costs),
        # Every page goes to an associate.
        "associate checks everything": three_way_cost(
            unsealed_scores, sealed_scores, 0.0, 1.01, **costs),
        # The model decides alone at its cheapest single cut.
        "model alone": best_cut,
    }


def staffing(review_share, pages_per_day=PAGES_PER_DAY, flagged_per_1000=0.0):
    """Associate hours a day for the review queue plus raising flagged pages."""
    reviewed = review_share * pages_per_day
    flagged = flagged_per_1000 / PER * pages_per_day
    hours = reviewed * REVIEW_SECONDS / 3600 + flagged * COST_FLAG / ASSOCIATE_RATE
    return {
        "reviewed_per_day": reviewed,
        "flagged_per_day": flagged,
        "hours_per_day": hours,
        "associates": hours / REVIEW_HOURS_PER_DAY,
    }
