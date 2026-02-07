"""
Audio file loader and preprocessor.
Python translation of read_sound_files.m

Reads WAV audio files from specified paths and extracts specific time segments
for analysis. Saves processed audio data to a pickle file for use by other scripts.
"""

import numpy as np
import librosa
import pickle


def read_sound_file(file_path, start_time_sec=None, end_time_sec=None):
    """
    Read audio file and optionally extract a time segment.

    Parameters:
    -----------
    file_path : str
        Path to the WAV file
    start_time_sec : float, optional
        Start time in seconds (if None, reads from beginning)
    end_time_sec : float, optional
        End time in seconds (if None, reads to end)

    Returns:
    --------
    audio : array
        Audio signal (mono)
    sample_rate : int
        Sample rate in Hz
    """
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=None, mono=True)

    # Extract time segment if specified
    if start_time_sec is not None and end_time_sec is not None:
        start_sample = int(start_time_sec * sample_rate)
        end_sample = int(end_time_sec * sample_rate)
        audio = audio[start_sample:end_sample]
    elif end_time_sec is not None:
        end_sample = int(end_time_sec * sample_rate)
        audio = audio[:end_sample]

    return audio, sample_rate


def main():
    """
    Main function to load all sound files and save them.

    Update the file paths below to match your local directory structure.
    """
    # Base directory for audio files
    # UPDATE THIS PATH to match your local directory
    base_dir = "data/"

    print("Loading audio files...")

    # Load File A: 15-12-22 B_with_sound_02.wav (22 minutes)
    print("Loading File A (15-12-22 B_with_sound_02.wav)...")
    A, sample_rate = read_sound_file(
        f"{base_dir}15-12-22 B_with_sound_02.wav",
        end_time_sec=22 * 60
    )

    # Load File B: 15-12-22_with_sound_02.wav (8 minutes)
    print("Loading File B (15-12-22_with_sound_02.wav)...")
    B, _ = read_sound_file(
        f"{base_dir}15-12-22_with_sound_02.wav",
        end_time_sec=8 * 60
    )

    # Load File C: 13-10-22 shotgun.wav (3 minutes 33 seconds)
    print("Loading File C (13-10-22 shotgun.wav)...")
    C, _ = read_sound_file(
        f"{base_dir}13-10-22 shotgun.wav",
        end_time_sec=3 * 60 + 33
    )

    # Load File D: 13-10-22 shotgun 2.wav (1 minute 50 seconds)
    print("Loading File D (13-10-22 shotgun 2.wav)...")
    D, _ = read_sound_file(
        f"{base_dir}13-10-22 shotgun 2.wav",
        end_time_sec=1 * 60 + 50
    )

    # Save all files to a pickle file
    sound_files = {
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'sample_rate': sample_rate
    }

    output_file = 'sound_files.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(sound_files, f)

    print(f"Sound files saved to {output_file}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"File A length: {len(A) / sample_rate:.2f} seconds")
    print(f"File B length: {len(B) / sample_rate:.2f} seconds")
    print(f"File C length: {len(C) / sample_rate:.2f} seconds")
    print(f"File D length: {len(D) / sample_rate:.2f} seconds")


if __name__ == "__main__":
    main()
