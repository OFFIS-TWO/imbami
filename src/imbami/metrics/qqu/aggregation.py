import numpy as np


def _region_mask(q: np.ndarray, g: np.ndarray | None, t: float | None) -> np.ndarray:
    """Membership mask of the cumulative-threshold region {i : g(y_i) >= t}."""
    if t is None:
        return np.ones_like(q, dtype=bool)
    if g is None:
        raise ValueError("g must be provided when t is specified.")
    g = np.asarray(g, dtype=float)
    if g.shape != q.shape:
        raise ValueError("g must have the same shape as q.")
    return g >= t


def QQU_avg(q: np.ndarray,
            g: np.ndarray | None = None,
            t: float | None = None,
            w: np.ndarray | None = None) -> float:
    """
    Regional QQU profile at a single threshold t (cumulative-threshold scheme).

    Implements QQU_avg(t) from the QQU framework (regional aggregation,
    Section "Regional aggregation"), using the cumulative-threshold
    membership function phi_t(y) = 1{g(y) >= t}:

        QQU_avg(t) = sum_i w_i * phi_t(y_i) * q_i
                     -----------------------------
                     sum_i w_i * phi_t(y_i)

    i.e. the w-weighted average of q over all samples whose domain
    position g(y_i) is at or beyond the threshold t.

    If t is None, no region is selected (phi_t = 1 for all samples) and
    the function reduces to the dataset-wide QQU_avg; g is then not
    needed and may be omitted.

    Parameters
    ----------
    q : np.ndarray, shape (N,)
        Per-sample QQU contributions q_i, corrected or not.
    g : np.ndarray, shape (N,), optional
        Per-sample domain position g(y_i); required only if t is given.
    t : float, optional
        Threshold defining the region {i : g(y_i) >= t}.
    w : np.ndarray, shape (N,), optional
        Per-sample weights w(y_i); None means uniform weighting.

    Returns
    -------
    float
        QQU_avg(t) over the selected region. NaN if the region is empty.
    """
    if w is None:
        w = np.ones_like(q, dtype=float)

    mask = _region_mask(q, g, t)
    denom = np.sum(w * mask)
    if denom == 0:
        return np.nan
    return np.sum(w * q * mask) / denom


def aggregate_c_delta(c_delta_i: np.ndarray,
                      *,
                      aggregator: str = "mean",
                      g: np.ndarray | None = None,
                      t: float | None = None,
                      w: np.ndarray | None = None) -> float:
    """
    Dataset-wide or regional aggregate of the sample-level corrections c_Delta,i.

    This is the aggregation counterpart of QQU_avg for the correction term, so
    that the corrected regional average can be read either as the weighted
    average of the corrected scores or as the difference of two aggregates:

        QQU_avg(q - c)_t = QQU_avg(q)_t - aggregate_c_delta(c, t).

    The correction of realized QQU is a constant, so its aggregation is trivial
    (mean = median = the constant). For expected QQU c_Delta,i varies across
    samples and the aggregate is what must be subtracted from QQU_avg(t).

    Parameters
    ----------
    c_delta_i : np.ndarray, shape (N,)
        Per-sample reference values, e.g. ``compute_c_delta(...).c_delta_i``.
    aggregator : {"mean", "median"}, default "mean"
        Must match the aggregator applied to the QQU scores downstream.
    g, t : optional
        Domain position and threshold selecting the region {g(y_i) >= t};
        t=None uses all samples.
    w : np.ndarray, shape (N,), optional
        Sample weights, used for aggregator="mean" only (sample weights are
        not supported for the median, as in QQU_median).

    Returns
    -------
    float
        Aggregate of c_Delta,i over the region. NaN if the region is empty.
    """
    c = np.asarray(c_delta_i, dtype=float)
    if c.ndim != 1:
        raise ValueError("c_delta_i must be a 1D array.")
    if aggregator == "median":
        mask = _region_mask(c, g, t)
        if not np.any(mask):
            return np.nan
        return float(np.median(c[mask]))
    if aggregator == "mean":
        return QQU_avg(c, g=g, t=t, w=w)
    raise ValueError(f"aggregator must be 'mean' or 'median', got {aggregator!r}")


def QQU_avg_profile(q: np.ndarray,
                     g: np.ndarray,
                     w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regional QQU profile QQU_avg(t), evaluated at every threshold induced by g.

    Vectorized computation of QQU_avg(t) across all distinct thresholds
    occurring in g, via a single sort and cumulative sum. Equivalent to calling
    QQU_avg(q, g, t, w) for every t in g, but O(N log N) instead of O(N^2).

    Also returns, at each threshold, the number of samples contributing to
    the average -- useful for judging the reliability of QQU_avg(t).

    Parameters
    ----------
    q : np.ndarray, shape (N,)
        Per-sample QQU contributions q_i, corrected or not.
    g : np.ndarray, shape (N,)
        Per-sample domain position g(y_i).
    w : np.ndarray, shape (N,), optional
        Per-sample weights w(y_i). If None (default), uniform weighting.

    Returns
    -------
    t_values : np.ndarray
        Distinct thresholds occurring in g, descending.
    qqu_avg_values : np.ndarray
        QQU_avg(t) evaluated at each threshold in t_values.
    n_values : np.ndarray
        Number of samples satisfying g(y_i) >= t at each threshold.
    """
    if w is None:
        w = np.ones_like(q, dtype=float)

    idx = np.argsort(g)[::-1]
    g_sorted = g[idx]
    w_sorted = w[idx]
    weighted_q_sorted = (w * q)[idx]

    qqu_sum_cum = np.cumsum(weighted_q_sorted)
    w_cum = np.cumsum(w_sorted)
    n_cum = np.arange(1, len(g_sorted) + 1)

    t_values, first_idx = np.unique(g_sorted, return_index=True)
    t_values = t_values[::-1]
    qqu_sum_values = qqu_sum_cum[first_idx][::-1]
    w_values = w_cum[first_idx][::-1]
    n_values = n_cum[first_idx][::-1]

    qqu_avg_values = qqu_sum_values / w_values

    return t_values, qqu_avg_values, n_values


def QQU_sum_profile(q: np.ndarray,
                     g: np.ndarray,
                     w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Unnormalized regional QQU sum, evaluated at every threshold induced by g.

    Vectorized computation of
        QQU_sum(t) = sum_i w_i * q_i * 1{g(y_i) >= t}
    for every distinct value t occurring in g. This is the numerator of
    QQU_avg(t), not itself a reported framework metric -- a building block for
    area-based / SERA-style summaries and for efficient plotting.

    Returns
    -------
    t_values : np.ndarray
        Distinct thresholds occurring in g, descending.
    qqu_sum_values : np.ndarray
        QQU_sum(t) evaluated at each threshold in t_values.
    """
    if w is None:
        w = np.ones_like(q, dtype=float)

    idx = np.argsort(g)[::-1]
    g_sorted = g[idx]
    weighted_q_sorted = (w * q)[idx]

    qqu_sum_cum = np.cumsum(weighted_q_sorted)

    t_values, first_idx = np.unique(g_sorted, return_index=True)
    t_values = t_values[::-1]
    qqu_sum_values = qqu_sum_cum[first_idx][::-1]

    return t_values, qqu_sum_values


def QQU_median(q: np.ndarray) -> float:
    """
    Dataset-wide median of the per-sample QQU contributions,
    QQU_median = median(q_1, ..., q_N).

    Less sensitive than QQU_avg to individual samples with a large
    discrepancy. Sample weights are not supported for the median.
    """
    return float(np.median(q))