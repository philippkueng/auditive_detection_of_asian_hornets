"""
Continuous Cross-correlation Product (CCP) analysis on test recordings.
Python translation of CCP.m

Applies the trained discriminant functions to analyze new audio recordings in a
sliding window fashion, computing discriminant function scores for each time segment.
"""

import numpy as np
import pickle
import librosa
from utils import two_D_FT_Gaussian


def analyze_audio_file(audio_file, start_min=0, start_sec=0, end_min=0, end_sec=0,
                        increment_sec=1.0):
    """
    Analyze an audio file using CCP with the trained discriminant functions.

    Parameters:
    -----------
    audio_file : str
        Path to audio file
    start_min : int
        Start time in minutes
    start_sec : float
        Start time seconds component
    end_min : int
        End time in minutes
    end_sec : float
        End time seconds component
    increment_sec : float
        Window increment in seconds (default 1.0)

    Returns:
    --------
    df_x : array
        DF score 1 for each window
    df_y : array
        DF score 2 for each window
    index_array : array
        Starting sample index for each window
    """
    print(f"Loading audio file: {audio_file}")

    # Load masking parameters (discriminant functions)
    with open('masking_parameters.pkl', 'rb') as f:
        params = pickle.load(f)

    new_dfa = params['new_dfa']
    new_dfa2 = params['new_dfa2']
    mf = params['mf']
    tr = params['tr']

    print(f"  Spectral repetition (mf): {mf} Hz")
    print(f"  Time resolution (tr): {tr} s")

    # Load audio file
    audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)

    # Calculate time range
    start_sample = int((start_min * 60 + start_sec) * sample_rate)

    # If end time is not specified (both 0), process entire file from start
    if end_min == 0 and end_sec == 0:
        end_sample = len(audio)
        print(f"  Processing entire file from {start_min}:{start_sec:02.0f}")
    else:
        end_sample = int((end_min * 60 + end_sec) * sample_rate)
        print(f"  Processing {start_min}:{start_sec:02.0f} to {end_min}:{end_sec:02.0f}")

    # Extract segment
    U = audio[start_sample:end_sample]
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Segment length: {len(U) / sample_rate:.2f} seconds")

    # Calculate increment in samples
    increment_shift = int(sample_rate * increment_sec)
    window_length = int(1.0 * sample_rate)  # 1 second windows

    print(f"  Window increment: {increment_sec} s ({increment_shift} samples)")
    print(f"  Window length: 1.0 s ({window_length} samples)")

    # Process audio in sliding windows
    df_scores = []
    index_array = []

    max_time = len(U) / (2 * sample_rate)

    print("\nProcessing audio windows...")
    n_windows = (len(U) - window_length) // increment_shift + 1
    print(f"  Total windows to process: {n_windows}")

    for i, start_time in enumerate(range(0, len(U) - window_length, increment_shift)):
        if i % 100 == 0:
            print(f"    Processing window {i+1}/{n_windows}...")

        # Extract audio section
        audio_section = U[start_time:start_time + window_length]

        # Compute 2D Fourier Transform
        tdft = np.abs(two_D_FT_Gaussian(audio_section, mf, tr, sample_rate, max_time))

        # Crop frequency range (4:60)
        tdft_cropped = tdft[4:60, :]

        # Scale by maximum
        max_tdft = np.max(tdft_cropped[:, 0])
        if max_tdft > 0:
            scaled_audio = tdft_cropped / max_tdft
        else:
            scaled_audio = tdft_cropped

        # Compute cross-correlation with discriminant functions
        ccp = np.sum(new_dfa * scaled_audio)
        ccp2 = np.sum(new_dfa2 * scaled_audio)

        df_scores.append([ccp, ccp2])
        index_array.append(start_time)

    df_scores = np.array(df_scores)
    df_x = df_scores[:, 0]
    df_y = df_scores[:, 1]
    index_array = np.array(index_array)

    print(f"\nProcessed {len(df_x)} windows")
    print(f"  DF score 1 range: [{df_x.min():.3f}, {df_x.max():.3f}]")
    print(f"  DF score 2 range: [{df_y.min():.3f}, {df_y.max():.3f}]")

    return df_x, df_y, index_array


def main():
    """
    Main function to analyze a test recording.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Analyze audio recording with CCP (Continuous Cross-correlation Product)'
    )
    parser.add_argument('audio_file', type=str, nargs='?', default='data/29-06-23.wav',
                        help='Path to audio file (default: data/29-06-23.wav)')
    parser.add_argument('--start-min', type=int, default=0,
                        help='Start time in minutes (default: 0)')
    parser.add_argument('--start-sec', type=float, default=0,
                        help='Start time seconds component (default: 0)')
    parser.add_argument('--end-min', type=int, default=0,
                        help='End time in minutes (default: 0 = entire file)')
    parser.add_argument('--end-sec', type=float, default=0,
                        help='End time seconds component (default: 0)')
    parser.add_argument('--increment', type=float, default=1.0,
                        help='Window increment in seconds (default: 1.0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file name (default: auto-generate from input)')

    args = parser.parse_args()

    print("Starting CCP analysis...")
    print("=" * 60)

    df_x, df_y, index_array = analyze_audio_file(
        args.audio_file,
        start_min=args.start_min,
        start_sec=args.start_sec,
        end_min=args.end_min,
        end_sec=args.end_sec,
        increment_sec=args.increment
    )

    # Save results
    if args.output:
        output_file = args.output
    else:
        # Generate output filename from input
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        output_file = f'{base_name}.pkl'

    results = {
        'df_x': df_x,
        'df_y': df_y,
        'index_array': index_array
    }

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
