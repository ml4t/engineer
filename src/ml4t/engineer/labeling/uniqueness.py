# mypy: disable-error-code="no-any-return,arg-type,call-arg,return-value,assignment"
"""Label uniqueness and sample weighting functions.

Implements De Prado's methods from AFML Chapter 4 for:
- Calculating label uniqueness based on overlapping periods
- Computing sample weights combining uniqueness and economic significance
- Sequential bootstrap for creating less redundant training sets

References
----------
.. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
       Chapter 4: Sample Weights.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, default_rng

from ml4t.engineer.labeling.numba_ops import _build_concurrency_nb


def build_concurrency(
    event_indices: npt.NDArray[np.float64],
    label_indices: npt.NDArray[np.float64],
    n_bars: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Calculate per-bar concurrency (how many labels are active at each time).

    This function computes c[t] = number of labels active at time t using
    an efficient O(n) difference-array algorithm.

    Parameters
    ----------
    event_indices : array
        Start indices of labels (when positions were entered)
    label_indices : array
        End indices of labels (when barriers were hit)
    n_bars : int, optional
        Total number of bars. If None, uses max(label_indices) + 1

    Returns
    -------
    array
        Concurrency at each timestamp (length = n_bars)

    Notes
    -----
    Concurrency is used to calculate label uniqueness. High concurrency
    at time t means many labels overlap there, indicating redundancy.

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 4: Sample Weights.

    Examples
    --------
    >>> concurrency = build_concurrency(event_indices, label_indices, len(prices))
    >>> # concurrency[t] = number of active labels at time t
    >>> max_overlap = concurrency.max()  # Maximum label overlap
    """
    if n_bars is None:
        n_bars = int(np.max(label_indices)) + 1

    return _build_concurrency_nb(n_bars, event_indices, label_indices)


# Keep old name as private alias for backward compatibility within the module
def _build_concurrency(
    n_bars: int,
    starts: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Legacy private function - use build_concurrency() instead."""
    return _build_concurrency_nb(n_bars, starts, ends)


def calculate_label_uniqueness(
    event_indices: npt.NDArray[np.intp],
    label_indices: npt.NDArray[np.intp],
    n_bars: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Calculate average uniqueness for each label based on overlapping periods.

    Uniqueness measures how "independent" a label is from others. Labels that
    overlap with many others have low uniqueness (redundant information), while
    labels that are relatively isolated have high uniqueness.

    Parameters
    ----------
    event_indices : array
        Start indices of labels (when positions were entered)
    label_indices : array
        End indices of labels (when barriers were hit)
    n_bars : int, optional
        Total number of bars. If None, uses max(label_indices) + 1

    Returns
    -------
    array
        Average uniqueness score for each label (between 0 and 1)

    Notes
    -----
    From López de Prado's AFML:
    u_i = (1/T_i) * Σ(1/c_t) for t in [start_i, end_i]

    Where:
    - T_i is the length of label i's active period
    - c_t is the concurrency at time t (number of active labels)
    - Higher uniqueness means more independent information

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 4: Sample Weights.
    """
    # Input validation
    if len(event_indices) != len(label_indices):
        raise ValueError(
            f"event_indices and label_indices must have same length, "
            f"got {len(event_indices)} and {len(label_indices)}"
        )

    if len(event_indices) == 0:
        return np.array([])

    if np.any(event_indices < 0) or np.any(label_indices < 0):
        raise ValueError("Indices must be non-negative")

    if n_bars is None:
        n_bars = int(np.max(label_indices)) + 1

    # Build concurrency array
    concurrency = _build_concurrency(
        n_bars,
        event_indices.astype(np.int64),
        label_indices.astype(np.int64),
    )

    # Calculate uniqueness for each label
    n_labels = len(event_indices)
    uniqueness = np.zeros(n_labels, dtype=np.float64)

    for i in range(n_labels):
        start = int(event_indices[i])
        end = int(label_indices[i])

        if start < n_bars and start <= end:
            # Ensure we don't go out of bounds
            start = max(0, start)
            end = min(end, n_bars - 1)

            # Average of 1/c_t over the label's active period
            c_slice = concurrency[start : end + 1]
            # Avoid division by zero (though concurrency should always be >= 1)
            uniqueness[i] = np.mean(1.0 / np.maximum(c_slice, 1.0))
        else:
            uniqueness[i] = 1.0  # Default for invalid ranges

    return uniqueness


def calculate_sample_weights(
    uniqueness: npt.NDArray[np.float64],
    returns: npt.NDArray[np.float64],
    weight_scheme: Literal[
        "returns_uniqueness", "uniqueness_only", "returns_only", "equal"
    ] = "returns_uniqueness",
) -> npt.NDArray[np.float64]:
    """
    Calculate sample weights combining statistical uniqueness and economic significance.

    Parameters
    ----------
    uniqueness : array
        Average uniqueness scores from calculate_label_uniqueness
    returns : array
        Label returns (from entry to exit)
    weight_scheme : str
        Weighting scheme to use:
        - "returns_uniqueness": u_i * |r_i| (De Prado's recommendation)
        - "uniqueness_only": u_i only (statistical correction)
        - "returns_only": |r_i| only (economic significance)
        - "equal": uniform weights

    Returns
    -------
    array
        Sample weights for training (normalized to sum to len(weights))

    Notes
    -----
    De Prado recommends "returns_uniqueness" to balance:
    - Statistical independence (uniqueness)
    - Economic importance (return magnitude)

    This prevents overweighting "boring" full-horizon labels while
    preserving the importance of profitable trades.

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 4: Sample Weights.
    """
    # Input validation
    if len(uniqueness) != len(returns):
        raise ValueError(
            f"uniqueness and returns must have same length, "
            f"got {len(uniqueness)} and {len(returns)}"
        )

    if len(uniqueness) == 0:
        return np.array([])

    if weight_scheme == "returns_uniqueness":
        # De Prado's formula: combine uniqueness with economic significance
        weights = uniqueness * np.abs(returns)
    elif weight_scheme == "uniqueness_only":
        weights = uniqueness
    elif weight_scheme == "returns_only":
        weights = np.abs(returns)
    else:  # "equal"
        weights = np.ones_like(uniqueness)

    # Normalize weights to sum to len(weights) for compatibility with ML libraries
    total = np.sum(weights)
    weights = weights * len(weights) / total if total > 0 else np.ones_like(uniqueness)

    return weights


def _expected_uniqueness_for_candidate(
    starts: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
    concurrency: npt.NDArray[np.int64],
    cand_idx: int,
) -> float:
    """
    Calculate expected uniqueness if we add candidate to the sample.
    u_j = mean_t(1 / (c_t + 1)) for t in [s_j, e_j]
    """
    s = int(starts[cand_idx])
    e = int(ends[cand_idx])
    if e < s:
        return 0.0

    c_slice = concurrency[s : e + 1]
    # +1 because we're calculating marginal uniqueness (if we add this label)
    return float(np.mean(1.0 / (c_slice + 1.0)))


def sequential_bootstrap(
    starts: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
    n_bars: int | None = None,
    n_draws: int | None = None,
    with_replacement: bool = True,
    random_state: int | Generator | None = None,
) -> npt.NDArray[np.int64]:
    """
    Sequential bootstrap that favors events with high marginal uniqueness.

    This method creates a bootstrapped sample that minimizes redundancy by
    probabilistically selecting labels based on how unique they would be
    given the already-selected labels.

    Parameters
    ----------
    starts : array
        Start indices of labels (event_indices)
    ends : array
        End indices of labels (label_indices)
    n_bars : int, optional
        Total number of bars. If None, uses max(ends) + 1
    n_draws : int, optional
        Number of selections to make. Defaults to len(starts)
    with_replacement : bool, default True
        If False, each event can be selected at most once
    random_state : int or Generator, optional
        RNG seed or Generator for reproducibility

    Returns
    -------
    array
        Indices of selected events in the order drawn (length = n_draws)

    Notes
    -----
    From López de Prado's AFML Chapter 4:
    - At each step, pick the event that maximizes expected average uniqueness
    - Probability of selection is proportional to marginal uniqueness
    - Creates less redundant training sets compared to random sampling

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 4: Sample Weights.

    Examples
    --------
    >>> # After triple barrier labeling
    >>> order = sequential_bootstrap(event_indices, label_indices, len(prices))
    >>> # Use order to select training samples
    >>> X_train = X[order]
    >>> y_train = y[order]
    >>> weights_train = sample_weights[order]
    """
    # Input validation
    if len(starts) != len(ends):
        raise ValueError(
            f"starts and ends must have same length, got {len(starts)} and {len(ends)}"
        )

    if len(starts) == 0:
        return np.array([], dtype=np.int64)

    if np.any(starts < 0) or np.any(ends < 0):
        raise ValueError("Indices must be non-negative")

    m = len(starts)
    if n_bars is None:
        n_bars = int(np.max(ends)) + 1
    if n_draws is None:
        n_draws = m

    if n_draws <= 0:
        raise ValueError(f"n_draws must be positive, got {n_draws}")

    rng: Generator = (
        default_rng(random_state) if not isinstance(random_state, Generator) else random_state
    )

    # Start with empty concurrency
    concurrency = np.zeros(n_bars, dtype=np.int64)
    available = np.ones(m, dtype=bool)  # track availability if sampling w/o replacement
    order = np.empty(n_draws, dtype=np.int64)

    for k in range(n_draws):
        # Compute marginal expected uniqueness for all available candidates
        u = np.zeros(m, dtype=np.float64)
        for j in range(m):
            if with_replacement or available[j]:
                u[j] = _expected_uniqueness_for_candidate(starts, ends, concurrency, j)
            else:
                u[j] = 0.0

        total = float(u.sum())
        if total <= 0.0:
            # Fallback to uniform over available items
            probs = np.where(available | with_replacement, 1.0, 0.0)
            prob_sum = probs.sum()
            if prob_sum == 0:
                # No valid candidates remaining
                raise ValueError(
                    f"Cannot draw {n_draws} samples without replacement from {m} candidates. "
                    f"Either reduce n_draws or set with_replacement=True."
                )
            probs = probs / prob_sum
        else:
            probs = u / total

        # Draw next index
        j = int(rng.choice(m, p=probs))
        order[k] = j

        # Update concurrency with the chosen interval
        s, e = int(starts[j]), int(ends[j])
        if s <= e and s < n_bars:
            s_clamped = max(0, min(s, n_bars - 1))
            e_clamped = max(0, min(e, n_bars - 1))
            concurrency[s_clamped : e_clamped + 1] += 1

        if not with_replacement:
            available[j] = False

    return order


__all__ = [
    "build_concurrency",
    "calculate_label_uniqueness",
    "calculate_sample_weights",
    "sequential_bootstrap",
]
