"""
Training Database (TDB) creation.
Python translation of TDB.m

Creates the training database by processing labeled audio samples and computing
2D Fourier Transforms for each category. This is the first step in the machine
learning pipeline.
"""

import numpy as np
import pickle
import librosa
import matplotlib.pyplot as plt
from utils import two_D_FT_Gaussian


def process_timings(audio, timings, window_length, sample_rate, mf, tr, max_time):
    """
    Process multiple time stamps and compute 2D Fourier Transforms.

    Parameters:
    -----------
    audio : array
        Audio signal
    timings : array
        Time stamps (in seconds) to process
    window_length : float
        Window length in seconds (±window_length around each timing)
    sample_rate : int
        Sample rate in Hz
    mf : float
        Spectral repetition parameter
    tr : float
        Time resolution
    max_time : float
        Maximum time for 2D FT

    Returns:
    --------
    tdft_array : 3D array
        Array of 2D Fourier Transforms (freq x spectral_rep x n_windows)
    """
    tdft_list = []

    for timing in timings:
        # Calculate window bounds
        lower_limit = int((timing - window_length) * sample_rate)
        upper_limit = int((timing + window_length) * sample_rate)

        # Ensure bounds are valid
        lower_limit = max(0, lower_limit)
        upper_limit = min(len(audio), upper_limit)

        # Extract window
        window = audio[lower_limit:upper_limit]

        # Compute 2D Fourier Transform
        tdft = two_D_FT_Gaussian(window, mf, tr, sample_rate, max_time)
        tdft_list.append(tdft)

    # Stack into 3D array
    if tdft_list:
        tdft_array = np.stack(tdft_list, axis=2)
    else:
        tdft_array = np.array([])

    return tdft_array


def scale_by_max(tdft_array, freq_range=(4, 60)):
    """
    Crop frequency range and scale by maximum value.

    Parameters:
    -----------
    tdft_array : 3D array
        Array of 2D Fourier Transforms
    freq_range : tuple
        (min, max) frequency indices to keep

    Returns:
    --------
    scaled_array : 3D array
        Scaled and cropped array
    """
    # Crop frequency range
    cropped = tdft_array[freq_range[0]:freq_range[1], :, :]

    # Scale each window by its maximum
    scaled_list = []
    for i in range(cropped.shape[2]):
        window = cropped[:, :, i]
        max_val = np.max(window[:, 0])  # Max in first column
        if max_val > 0:
            scaled = window / max_val
        else:
            scaled = window
        scaled_list.append(scaled)

    scaled_array = np.stack(scaled_list, axis=2)
    return scaled_array


def main():
    """
    Main function to create training database.
    """
    print("Loading sound files...")

    # Load preprocessed sound files
    with open('sound_files.pkl', 'rb') as f:
        sound_files = pickle.load(f)

    A = sound_files['A']
    B = sound_files['B']
    C = sound_files['C']
    D = sound_files['D']
    sample_rate = sound_files['sample_rate']

    # Also load file E: 16-05-23.wav (5 minutes 20 seconds)
    print("Loading additional file E (16-05-23.wav)...")
    E, _ = librosa.load('data/16-05-23.wav', sr=sample_rate, mono=True)
    E = E[:int((5 * 60 + 20) * sample_rate)]

    # Processing parameters
    window_length = 0.5  # seconds
    mf = 4  # spectral repetition (Hz)
    tr = 0.04  # time resolution (seconds)
    max_time = len(A) / (2 * sample_rate)

    print("Processing parameters:")
    print(f"  Window length: {window_length} s")
    print(f"  Spectral repetition (MF): {mf} Hz")
    print(f"  Time resolution (tr): {tr} s")

    # Define timings for each category (in seconds)

    # Irregular hornet, not detected (A)
    timings_IHND = np.array([21, 67, 75, 81, 91, 171, 179, 208, 221, 232, 239,
                              248, 253, 506, 520, 523, 532, 539, 555, 563, 882])

    # Irregular hornet, detected (A)
    timings_IHD = np.array([15, 37, 63, 85, 197, 204, 217, 225, 263, 269, 442,
                             510, 515, 526, 536, 1254, 1279.5])

    # Regular hornet, always detected (C)
    timings_RHD1 = np.array([3, 4, 5, 9, 10, 17, 18, 40, 41, 62, 68, 70, 72, 79,
                              85, 91, 98, 107.5, 112, 118, 120, 155, 163, 166,
                              172, 185, 199])

    # Regular hornet, always detected (D)
    timings_RHD2 = np.array([3, 4, 5, 6, 7, 10, 11, 13, 14, 15, 22, 39, 41, 46,
                              47, 59, 60, 61, 64, 70, 85, 92, 102])

    # Bees detected as hornet (B)
    timings_BH = np.array([83, 91, 93, 98, 114, 123, 124, 128, 131, 141, 174,
                            208, 211, 216, 237, 265.5, 319, 323, 338, 341.5,
                            350, 353, 355, 360, 361, 362, 380, 391, 400, 401,
                            414, 417])

    # Bees not detected as hornet (A)
    timings_BEE = np.array([136, 148, 157, 167, 307.5, 309, 310, 312, 329, 357.5,
                             366, 368, 373, 385, 388, 395, 427, 477, 493, 500,
                             579, 590, 631, 634, 638, 667])

    # 16.05.23 bees (E)
    timings_BEE16 = np.array([8.5, 12.7, 19, 20.7, 31, 39.5, 42, 46, 49.5, 52,
                               57, 61, 65, 67.5, 71.5, 75, 80.5, 82.5, 84.5, 91,
                               93, 96, 98, 100.5, 106, 119.5, 123] +
                              list(range(130, 134)) +
                              [134.5, 135.5, 141, 146, 149, 153, 154, 155.5,
                               168, 171.5, 174.4, 182, 185, 193, 200.5, 203,
                               211, 215.5, 229, 238, 242, 251.5, 254])

    # Background (B)
    timings_BGB = np.array([52, 53] + list(range(61, 64)) + [101, 102, 109, 117,
                            147, 150, 162] + list(range(218, 234)) +
                            list(range(243, 257)) + list(range(274, 279)) +
                            list(range(410, 413)) + list(range(433, 437)) +
                            list(range(450, 457)) + list(range(459, 464)))

    # Background (C)
    timings_BGC = np.array(list(range(35, 38)) + list(range(44, 52)) +
                            [142, 143, 202, 203])

    # 16.05.23 background (E)
    timings_BGE = np.array([6, 25.5, 33, 35.5, 37, 43.5, 47, 74, 78.7, 96.7, 99,
                             102, 104, 237, 266, 279, 282, 288.5, 305.5, 310])

    print("\nProcessing audio segments...")

    # Process each category
    print("  Processing IHND (Irregular Hornet Not Detected)...")
    IHND_tdft = process_timings(A, timings_IHND, window_length, sample_rate,
                                 mf, tr, max_time)

    print("  Processing IHD (Irregular Hornet Detected)...")
    IHD_tdft = process_timings(A, timings_IHD, window_length, sample_rate,
                                mf, tr, max_time)

    print("  Processing RHD1 (Regular Hornet Detected 1)...")
    RHD1_tdft = process_timings(C, timings_RHD1, window_length, sample_rate,
                                 mf, tr, max_time)

    print("  Processing RHD2 (Regular Hornet Detected 2)...")
    RHD2_tdft = process_timings(D, timings_RHD2, window_length, sample_rate,
                                 mf, tr, max_time)

    print("  Processing BH (Bees detected as Hornet)...")
    BH_tdft = process_timings(B, timings_BH, window_length, sample_rate,
                               mf, tr, max_time)

    print("  Processing BEE (Bees)...")
    BEE_tdft = process_timings(A, timings_BEE, window_length, sample_rate,
                                mf, tr, max_time)

    print("  Processing BEE16 (Bees from file E)...")
    BEE16_tdft = process_timings(E, timings_BEE16, window_length, sample_rate,
                                  mf, tr, max_time)

    print("  Processing BGB (Background from B)...")
    BGB_tdft = process_timings(B, timings_BGB, window_length, sample_rate,
                                mf, tr, max_time)

    print("  Processing BGC (Background from C)...")
    BGC_tdft = process_timings(C, timings_BGC, window_length, sample_rate,
                                mf, tr, max_time)

    print("  Processing BGE (Background from E)...")
    BGE_tdft = process_timings(E, timings_BGE, window_length, sample_rate,
                                mf, tr, max_time)

    # Visualize an example 2DFT
    print("\nGenerating example 2DFT visualization...")
    if BEE16_tdft.shape[2] > 29:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(np.log10(BEE16_tdft[:, :, 29]),
                       extent=[0, 0.5 * mf / tr, 0, sample_rate / 2],
                       aspect='auto', origin='lower', cmap='jet')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1500)
        ax.set_xlabel('Spectral repetition (Hz)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Example 2D Fourier Transform (BEE16, window 30)')
        plt.colorbar(im, ax=ax, label='Acceleration magnitude (m/s²)')
        plt.tight_layout()
        plt.savefig('example_2dft.png', dpi=150)
        print("  Saved example_2dft.png")

    # Scale by maximum and crop frequency range
    print("\nScaling spectra by maximum values...")
    scaled_IHND = scale_by_max(IHND_tdft)
    scaled_IHD = scale_by_max(IHD_tdft)
    scaled_RHD1 = scale_by_max(RHD1_tdft)
    scaled_RHD2 = scale_by_max(RHD2_tdft)
    scaled_BH = scale_by_max(BH_tdft)
    scaled_BEE = scale_by_max(BEE_tdft)
    scaled_BEE16 = scale_by_max(BEE16_tdft)
    scaled_BGB = scale_by_max(BGB_tdft)
    scaled_BGC = scale_by_max(BGC_tdft)
    scaled_BGE = scale_by_max(BGE_tdft)

    # Combine categories
    print("\nCombining categories...")
    scaled_hornet = np.concatenate([scaled_IHND, scaled_IHD, scaled_RHD1,
                                     scaled_RHD2], axis=2)
    scaled_bee = np.concatenate([scaled_BH, scaled_BEE, scaled_BEE16], axis=2)
    scaled_BG = np.concatenate([scaled_BGB, scaled_BGC], axis=2)
    # scaled_BGE remains separate (summer background)

    # Save training database
    tdb_data = {
        'scaled_hornet': scaled_hornet,
        'scaled_bee': scaled_bee,
        'scaled_BG': scaled_BG,
        'scaled_BGE': scaled_BGE,
        'sample_rate': sample_rate,
        'mf': mf,
        'tr': tr,
        'window_length': window_length
    }

    output_file = 'fourth_TDB.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(tdb_data, f)

    print(f"\nTraining database saved to {output_file}")
    print(f"  Hornet samples: {scaled_hornet.shape[2]}")
    print(f"  Bee samples: {scaled_bee.shape[2]}")
    print(f"  Winter background samples: {scaled_BG.shape[2]}")
    print(f"  Summer background samples: {scaled_BGE.shape[2]}")
    print(f"  Spectrum shape: {scaled_hornet.shape[0]} x {scaled_hornet.shape[1]}")


if __name__ == "__main__":
    main()
