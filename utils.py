"""
Utility functions for hornet detection system.
Python translation of custom MATLAB functions.
"""

import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def two_D_FT_Gaussian(audio_signal, mf, tr, sample_rate, max_time):
    """
    Compute 2D Fourier Transform using Gaussian window.

    This uses a spectrogram-based approach for efficiency:
    1. Compute STFT (Short-Time Fourier Transform) to get time-frequency representation
    2. Apply FFT along time axis for each frequency to get spectral repetitions

    Parameters:
    -----------
    audio_signal : array
        Audio signal to process
    mf : float
        Spectral repetition parameter (Hz)
    tr : float
        Time resolution (seconds)
    sample_rate : float
        Sample rate of audio (Hz)
    max_time : float
        Maximum time for analysis

    Returns:
    --------
    tdft : 2D array
        2D Fourier Transform (frequency x spectral repetition)
    """
    # Ensure audio_signal is 1D
    if len(audio_signal.shape) > 1:
        audio_signal = audio_signal.flatten()

    # Window parameters
    nperseg = int(tr * sample_rate)  # Window length in samples
    noverlap = int(nperseg * 0.75)  # 75% overlap

    # Create Gaussian window
    window = signal.windows.gaussian(nperseg, std=nperseg/6)

    # Compute spectrogram using STFT
    frequencies, times, Zxx = signal.stft(
        audio_signal,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False
    )

    # Zxx is (n_freqs x n_times), complex values
    # Take magnitude
    spectrogram = np.abs(Zxx)

    # Compute modulation spectrum (spectral repetition) for each frequency
    # by taking FFT along time axis
    n_freqs = spectrogram.shape[0]
    n_times = spectrogram.shape[1]

    # Apply FFT along time axis
    modulation_spectrum = np.fft.fft(spectrogram, axis=1)

    # Get modulation frequencies
    modulation_freqs = np.fft.fftfreq(n_times, d=1.0/sample_rate * nperseg)

    # Only keep positive frequencies up to max_sr
    max_sr = 0.5 * mf / tr
    positive_idx = modulation_freqs >= 0
    modulation_freqs = modulation_freqs[positive_idx]
    modulation_spectrum = modulation_spectrum[:, positive_idx]

    # Limit to max_sr
    valid_idx = modulation_freqs <= max_sr
    modulation_freqs = modulation_freqs[valid_idx]
    tdft = np.abs(modulation_spectrum[:, valid_idx])

    return tdft


def dfa(data, labels, n_components=2):
    """
    Discriminant Function Analysis using Linear Discriminant Analysis.

    Parameters:
    -----------
    data : array, shape (n_samples, n_features)
        Input data
    labels : array, shape (n_samples,)
        Class labels
    n_components : int
        Number of discriminant functions to compute

    Returns:
    --------
    U : array
        Discriminant function scores
    V : array
        Discriminant function coefficients
    eigenval : array
        Eigenvalues
    """
    # Use sklearn's LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    # Fit and transform
    U = lda.fit_transform(data, labels)

    # Get coefficients (scalings)
    V = lda.scalings_

    # Get explained variance ratio as proxy for eigenvalues
    eigenval = lda.explained_variance_ratio_

    return U, V, eigenval


def PCA_deviations(decay_curve, n_components):
    """
    Plot PCA decay curve showing deviation of eigenvalues.

    Parameters:
    -----------
    decay_curve : array
        PCA eigenvalue decay curve
    n_components : int
        Number of components to highlight
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(decay_curve, 'bo-')
    ax.axvline(x=n_components, color='r', linestyle='--',
               label=f'{n_components} components')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue Magnitude')
    ax.set_title('PCA Decay Curve')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def create_polygon_boundary(points, shrink_factor=0.1):
    """
    Create a boundary polygon around a set of points.

    Parameters:
    -----------
    points : array, shape (n_points, 2)
        2D points (x, y coordinates)
    shrink_factor : float
        Boundary shrink factor (0-1, lower = tighter boundary)

    Returns:
    --------
    boundary_points : array
        Points defining the boundary polygon
    """
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return points

    try:
        # Compute convex hull
        hull = ConvexHull(points)
        boundary_indices = hull.vertices
        boundary_points = points[boundary_indices]

        # Apply shrinking toward centroid if needed
        if shrink_factor > 0:
            centroid = np.mean(points, axis=0)
            boundary_points = centroid + (1 - shrink_factor) * (boundary_points - centroid)

        return boundary_points
    except:
        # If convex hull fails, return all points
        return points


def points_in_polygon(test_points, polygon):
    """
    Determine if points are inside a polygon.

    Parameters:
    -----------
    test_points : array, shape (n_points, 2)
        Points to test
    polygon : array, shape (n_vertices, 2)
        Polygon vertices

    Returns:
    --------
    inside : array, shape (n_points,)
        Boolean array indicating if each point is inside
    """
    from matplotlib.path import Path

    path = Path(polygon)
    inside = path.contains_points(test_points)
    return inside
