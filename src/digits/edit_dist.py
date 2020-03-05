# 13/01/2020
# implementation of the edit distance comparison used by Seniuk

from scipy.signal import argrelextrema
import numpy as np
from numba import jit


def get_peaks(signal, sr):
    # Raw maxima, without peak rejections
    # (found by Seniuk to perform the best)
    peaks_idx = argrelextrema(signal, np.greater)[0]
    peaks_times = peaks_idx / sr
    return np.array([peaks_times, signal[peaks_idx]]).T


@jit(nopython=True)
def sub_cost_mult(peak1, peak2, p=1):
    """Multiplicative substitution cost (found by Seniuk to outperform an additive cost function). Weighted product of difference in time and difference in height."""
    # Confusion: should be zero when peak1 == peak2
    t1, h1 = peak1
    t2, h2 = peak2
    delta_t = abs(t1 - t2)
    delta_h = abs(h1 - h2)
    return (delta_t ** p) * (delta_h ** (1 - p))


def edit_distance(peaks1, peaks2, sub_cost):
    """Computes edit distance between peaks1 and peaks2.
Insertion/deletion cost is the height of the peak.
Substitution cost is calculated using the function sub_cost"""
    m, n = len(peaks1), len(peaks2)
    d = [[0 for i in range(n + 1)] for j in range(m + 1)]
    for i in range(m):
        d[i+1][0] = d[i][0] + peaks1[i][1]
    for j in range(n):
        d[0][j+1] = d[0][j] + peaks2[j][1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            insertion_cost = d[i-1][j] + peaks1[i-1][1]
            deletion_cost = d[i][j-1] + peaks2[j-1][1]
            substitution_cost = d[i-1][j-1] + sub_cost(peaks1[i-1], peaks2[j-1])
            d[i][j] = min(insertion_cost, deletion_cost, substitution_cost)

    return d[m][n]


def test_edit_distance():
    test1 = np.array([1, 7, 3, 4, 5, 4, 7, 4, 5, 4])
    test2 = np.array([1, 3, 5, 4, 5, 3, 5, 3, 2, 4])
    peaks1 = get_peaks(test1, 10)
    peaks2 = get_peaks(test2, 10)
    sub_cost_func = lambda p1, p2: sub_cost_mult(p1, p2, p=1.0)
    assert edit_distance(peaks1, peaks2, sub_cost_func) == 5.1