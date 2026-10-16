import numpy as np



def QQU_avg(q: np.ndarray,
            g: np.ndarray | None = None,
            t: float| None = None,
            w: np.ndarray |None = None) -> float:
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
        Per-sample QQU contributions q_i (either q_i^R or q_i^E; this
        function is agnostic to which).
    g : np.ndarray, shape (N,), optional
        Per-sample domain position g(y_i) used to select the region
        (e.g. a relevance function, or the target value itself).
        Required only if t is given.
    t : float, optional
        Threshold defining the region {i : g(y_i) >= t}. If None
        (default), no restriction is applied, recovering the
        dataset-wide QQU_avg.
    w : np.ndarray, shape (N,), optional
        Per-sample weights w(y_i). If None (default), uniform weighting
        w_i = 1 is used.

    Returns
    -------
    float
        QQU_avg(t): the weighted-average QQU over the selected region.
        NaN if no sample satisfies g(y_i) >= t.
    """
    if w is None:
        w = np.ones_like(q, dtype=float)

    if t is None:
        mask = np.ones_like(q, dtype=bool)
    else:
        if g is None:
            raise ValueError("g must be provided when t is specified.")
        mask = g >= t

    denom = np.sum(w * mask)
    if denom == 0:
        return np.nan
    return np.sum(w * q * mask) / denom






def QQU_avg_profile(q: np.ndarray,
                     g: np.ndarray,
                     w: np.ndarray|None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regional QQU profile QQU_avg(t), evaluated at every threshold induced by g.

    Vectorized computation of QQU_avg(t) = QQU_sum(t) / sum_i w_i * 1{g(y_i) >= t}
    (see QQU_avg for the single-threshold definition) across all distinct
    thresholds occurring in g, via a single sort and cumulative sum.
    Equivalent to calling QQU_avg(q, g, t, w) for every t in g, but O(N log N)
    instead of O(N^2).

    Also returns, at each threshold, the number of samples contributing to
    the average -- useful for judging the reliability of QQU_avg(t), since
    the profile becomes noisier as the contributing sample count shrinks
    towards the tails of g.

    Parameters
    ----------
    q : np.ndarray, shape (N,)
        Per-sample QQU contributions q_i.
    g : np.ndarray, shape (N,)
        Per-sample domain position g(y_i).
    w : np.ndarray, shape (N,), optional
        Per-sample weights w(y_i). If None (default), uniform weighting
        w_i = 1 is used.

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
                     w: np.ndarray | None = None)-> tuple[np.ndarray, np.ndarray]:
    """
    Unnormalized regional QQU sum, evaluated at every threshold induced by g.

    Vectorized computation of
        QQU_sum(t) = sum_i w_i * q_i * 1{g(y_i) >= t}
    for every distinct value t occurring in g, via a single sort and
    cumulative sum. This is the numerator of QQU_avg(t) (see QQU_avg),
    not itself a reported framework metric -- it is intended as a
    building block (e.g. for area-based / SERA-style summaries and
    for efficient plotting over many thresholds at once.

    Parameters
    ----------
    q : np.ndarray, shape (N,)
        Per-sample QQU contributions q_i.
    g : np.ndarray, shape (N,)
        Per-sample domain position g(y_i).
    w : np.ndarray, shape (N,), optional
        Per-sample weights w(y_i). If None (default), uniform weighting
        w_i = 1 is used.

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