"""
Feature extraction and discriminant function generation.
Python translation of PCA_DFA.m

Performs Principal Component Analysis and Discriminant Function Analysis on the
training database to create discriminant spectra for classification.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import PCA_deviations, dfa


def reshape_to_vectors(data_3d):
    """
    Reshape 3D array (freq x spectral_rep x n_samples) to 2D array (n_samples x features).

    Parameters:
    -----------
    data_3d : array, shape (n_freq, n_sr, n_samples)
        3D array of spectra

    Returns:
    --------
    data_2d : array, shape (n_samples, n_features)
        Flattened spectra as row vectors
    """
    n_samples = data_3d.shape[2]
    data_list = []

    for i in range(n_samples):
        spectrum = data_3d[:, :, i]
        flattened = spectrum.flatten()
        data_list.append(flattened)

    data_2d = np.array(data_list)
    return data_2d


def main():
    """
    Main function to perform PCA and DFA on training database.
    """
    print("Loading training database...")

    # Load training database
    with open('fourth_TDB.pkl', 'rb') as f:
        tdb_data = pickle.load(f)

    scaled_hornet = tdb_data['scaled_hornet']
    scaled_bee = tdb_data['scaled_bee']
    scaled_BG = tdb_data['scaled_BG']
    scaled_BGE = tdb_data['scaled_BGE']
    sample_rate = tdb_data['sample_rate']
    mf = tdb_data['mf']
    tr = tdb_data['tr']

    print(f"  Hornet samples: {scaled_hornet.shape[2]}")
    print(f"  Bee samples: {scaled_bee.shape[2]}")
    print(f"  Winter background samples: {scaled_BG.shape[2]}")
    print(f"  Summer background samples: {scaled_BGE.shape[2]}")

    # Convert to 2D arrays (samples x features)
    print("\nReshaping data...")
    hornet_array = reshape_to_vectors(scaled_hornet)
    bee_array = reshape_to_vectors(scaled_bee)
    BG_array = reshape_to_vectors(scaled_BG)
    summerBG_array = reshape_to_vectors(scaled_BGE)

    # Combine all data
    TDB = np.vstack([hornet_array, bee_array, BG_array, summerBG_array])

    # Visualize training database
    print("\nGenerating training database visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))
    vmin = np.log10(np.maximum(TDB.max() / 300, 1e-10))
    vmax = np.log10(TDB.max())
    im = ax.imshow(np.log10(np.maximum(TDB, 1e-10)), aspect='auto',
                   cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Data points per pulse', fontsize=14)
    ax.set_ylabel('Pulse number', fontsize=14)
    ax.set_title('Training database for bee & hornet discrimination', fontsize=16)
    plt.colorbar(im, ax=ax, label='Acceleration magnitude (m/s²)')
    plt.tight_layout()
    plt.savefig('training_database.png', dpi=150)
    print("  Saved training_database.png")

    # Get sample counts
    hornet_wavs = scaled_hornet.shape[2]
    bee_wavs = scaled_bee.shape[2]
    bg_wavs = scaled_BG.shape[2]
    bgs_wavs = scaled_BGE.shape[2]

    # Perform PCA
    print("\nPerforming PCA...")
    # Center the data
    mean_spectrum = np.mean(TDB, axis=0)
    centered_data = TDB - mean_spectrum

    # Compute PCA using sklearn
    pca = PCA()
    scores = pca.fit_transform(centered_data)
    eigenspectra = pca.components_  # Each row is a principal component

    # Visualize PCA scores (first 2 components)
    print("\nGenerating PCA visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(scores[:hornet_wavs, 0], scores[:hornet_wavs, 1],
            'ro', markerfacecolor='r', markersize=6, label='Hornet')
    ax.plot(scores[hornet_wavs:hornet_wavs+bee_wavs, 0],
            scores[hornet_wavs:hornet_wavs+bee_wavs, 1],
            'bo', markerfacecolor='b', markersize=6, label='Bee')
    ax.plot(scores[hornet_wavs+bee_wavs:hornet_wavs+bee_wavs+bg_wavs, 0],
            scores[hornet_wavs+bee_wavs:hornet_wavs+bee_wavs+bg_wavs, 1],
            'ko', markerfacecolor='k', markersize=6, label='Winter background')
    ax.plot(scores[hornet_wavs+bee_wavs+bg_wavs:, 0],
            scores[hornet_wavs+bee_wavs+bg_wavs:, 1],
            'co', markerfacecolor='c', markersize=6, label='Summer background')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('PCA Scores')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('pca_scores.png', dpi=150)
    print("  Saved pca_scores.png")

    # Perform DFA with first 9 principal components
    print("\nPerforming DFA...")
    limit = 9
    pca_scores = scores[:, :limit]

    # Create labels: 0=hornet, 1=bee, 2=winter bg, 3=summer bg
    labels = np.array([0] * hornet_wavs + [1] * bee_wavs +
                      [2] * bg_wavs + [3] * bgs_wavs)

    # Apply DFA (using custom function from utils)
    U, V, eigenval = dfa(pca_scores, labels, n_components=2)

    # Compute discriminant spectra
    # DFA_spectrum = sum of (eigenspectra * DFA coefficients)
    DFA_spectrum_01 = np.sum(eigenspectra[:limit, :].T * V[:, 0], axis=1)
    DFA_spectrum_02 = np.sum(eigenspectra[:limit, :].T * V[:, 1], axis=1)

    # Compute DF scores for all training data
    validation_data = TDB
    A_x = np.sum(validation_data * DFA_spectrum_01, axis=1)
    A_y = np.sum(validation_data * DFA_spectrum_02, axis=1)

    # Reshape discriminant spectra back to 2D
    spectrum_shape = (scaled_hornet.shape[0], scaled_hornet.shape[1])
    new_dfa = DFA_spectrum_01.reshape(spectrum_shape)
    new_dfa2 = DFA_spectrum_02.reshape(spectrum_shape)

    # Calculate centroids
    centroid_01 = [np.mean(A_x[:hornet_wavs]), np.mean(A_y[:hornet_wavs])]
    centroid_02 = [np.mean(A_x[hornet_wavs:hornet_wavs+bee_wavs]),
                   np.mean(A_y[hornet_wavs:hornet_wavs+bee_wavs])]
    centroid_03 = [np.mean(A_x[hornet_wavs+bee_wavs:hornet_wavs+bee_wavs+bg_wavs]),
                   np.mean(A_y[hornet_wavs+bee_wavs:hornet_wavs+bee_wavs+bg_wavs])]
    centroid_04 = [np.mean(A_x[hornet_wavs+bee_wavs+bg_wavs:]),
                   np.mean(A_y[hornet_wavs+bee_wavs+bg_wavs:])]

    # Visualize DFA outcome
    print("\nGenerating DFA visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(A_x[:hornet_wavs], A_y[:hornet_wavs],
            'ro', markerfacecolor='r', markersize=6, label='Hornet')
    ax.plot(A_x[hornet_wavs:hornet_wavs+bee_wavs],
            A_y[hornet_wavs:hornet_wavs+bee_wavs],
            'bo', markerfacecolor='b', markersize=6, label='Bee')
    ax.plot(A_x[hornet_wavs+bee_wavs:hornet_wavs+bee_wavs+bg_wavs],
            A_y[hornet_wavs+bee_wavs:hornet_wavs+bee_wavs+bg_wavs],
            'ko', markerfacecolor='k', markersize=6, label='Winter background')
    ax.plot(A_x[hornet_wavs+bee_wavs+bg_wavs:],
            A_y[hornet_wavs+bee_wavs+bg_wavs:],
            'co', markerfacecolor='c', markersize=6, label='Summer background')

    # Plot centroids
    ax.plot(centroid_01[0], centroid_01[1], 'yo', markersize=15, linewidth=3,
            markeredgecolor='black', label='Hornet centroid')
    ax.plot(centroid_02[0], centroid_02[1], 'go', markersize=15, linewidth=3,
            markeredgecolor='black', label='Bee centroid')
    ax.plot(centroid_03[0], centroid_03[1], 'mo', markersize=15, linewidth=3,
            markeredgecolor='black', label='Winter BG centroid')
    ax.plot(centroid_04[0], centroid_04[1], 'ko', markersize=15, linewidth=3,
            markeredgecolor='white', label='Summer BG centroid')

    ax.set_xlabel('DF score 1', fontsize=14)
    ax.set_ylabel('DF score 2', fontsize=14)
    ax.set_title('DFA outcome (scatterplot)', fontsize=16)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('dfa_outcome.png', dpi=150)
    print("  Saved dfa_outcome.png")

    # Visualize discriminant spectra
    print("\nGenerating discriminant spectra visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vmin1 = np.log10(np.maximum(np.abs(new_dfa).max() / 300, 1e-10))
    vmax1 = np.log10(np.abs(new_dfa).max())
    im1 = ax1.imshow(np.log10(np.maximum(np.abs(new_dfa), 1e-10)),
                     extent=[0, 49, 0, 1500], aspect='auto', origin='lower',
                     cmap='jet', vmin=vmin1, vmax=vmax1)
    ax1.set_xlabel('Spectral Repetition (Hz)', fontsize=14)
    ax1.set_ylabel('Frequency (Hz)', fontsize=14)
    ax1.set_title('DF spectrum 1', fontsize=16)
    plt.colorbar(im1, ax=ax1, label='Acceleration magnitude (a.u.)')

    vmin2 = np.log10(np.maximum(np.abs(new_dfa2).max() / 300, 1e-10))
    vmax2 = np.log10(np.abs(new_dfa2).max())
    im2 = ax2.imshow(np.log10(np.maximum(np.abs(new_dfa2), 1e-10)),
                     extent=[0, 49, 0, 1500], aspect='auto', origin='lower',
                     cmap='jet', vmin=vmin2, vmax=vmax2)
    ax2.set_xlabel('Spectral Repetition (Hz)', fontsize=14)
    ax2.set_ylabel('Frequency (Hz)', fontsize=14)
    ax2.set_title('DF spectrum 2', fontsize=16)
    plt.colorbar(im2, ax=ax2, label='Acceleration magnitude (a.u.)')

    plt.tight_layout()
    plt.savefig('df_spectra.png', dpi=150)
    print("  Saved df_spectra.png")

    # Plot PCA decay curve
    print("\nGenerating PCA decay curve...")
    decay_curve = np.mean(np.abs(scores), axis=0)
    fig = PCA_deviations(decay_curve, 6)
    plt.savefig('pca_decay.png', dpi=150)
    print("  Saved pca_decay.png")

    # Save masking parameters
    masking_params = {
        'new_dfa': new_dfa,
        'new_dfa2': new_dfa2,
        'mf': mf,
        'tr': tr,
        'centroid_01': centroid_01,
        'centroid_02': centroid_02,
        'centroid_03': centroid_03,
        'centroid_04': centroid_04,
        'A_x': A_x,
        'A_y': A_y,
        'hornet_wavs': hornet_wavs,
        'bee_wavs': bee_wavs,
        'bg_wavs': bg_wavs,
        'bgs_wavs': bgs_wavs
    }

    output_file = 'masking_parameters.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(masking_params, f)

    print(f"\nMasking parameters saved to {output_file}")
    print("\nCentroids:")
    print(f"  Hornet: {centroid_01}")
    print(f"  Bee: {centroid_02}")
    print(f"  Winter background: {centroid_03}")
    print(f"  Summer background: {centroid_04}")


if __name__ == "__main__":
    main()
