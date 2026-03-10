"""
Extract 10-second audio segments containing bee or hornet sounds from shotgun files.

This script:
1. Loads shotgun audio files from the data directory
2. Uses the trained ML classifier to identify bee/hornet sounds in 1-second windows
3. Extracts 10-second segments containing detected sounds
4. Saves each segment as an individual WAV file with labels
"""

import numpy as np
import pickle
import librosa
import soundfile as sf
import os
from pathlib import Path
from utils import two_D_FT_Gaussian


def load_trained_model():
    """Load the trained ML classifier."""
    print("Loading trained ML model...")
    with open('trained_ml_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)

    print(f"  Classifier: {model_data['classifier_name']}")
    print(f"  Classes: {model_data['class_names']}")
    return model_data


def load_masking_parameters():
    """Load masking parameters for 2D FT computation."""
    with open('masking_parameters.pkl', 'rb') as f:
        params = pickle.load(f)
    return params['mf'], params['tr']


def extract_features_from_window(audio_window, mf, tr, sample_rate):
    """
    Extract 2D FT features from a 1-second audio window.

    Parameters:
    -----------
    audio_window : array
        Audio samples (1 second)
    mf : float
        Spectral repetition parameter
    tr : float
        Time resolution
    sample_rate : int
        Sample rate in Hz

    Returns:
    --------
    feature_vector : array
        Flattened feature vector
    """
    max_time = len(audio_window) / (2 * sample_rate)

    # Compute 2D Fourier Transform
    tdft = np.abs(two_D_FT_Gaussian(audio_window, mf, tr, sample_rate, max_time))

    # Crop frequency range (4:60)
    tdft_cropped = tdft[4:60, :]

    # Scale by maximum
    max_tdft = np.max(tdft_cropped[:, 0])
    if max_tdft > 0:
        scaled_audio = tdft_cropped / max_tdft
    else:
        scaled_audio = tdft_cropped

    # Flatten to feature vector
    return scaled_audio.flatten()


def classify_audio_windows(audio, sample_rate, model_data, mf, tr, window_sec=1.0, increment_sec=0.5):
    """
    Classify audio in sliding windows.

    Parameters:
    -----------
    audio : array
        Audio signal
    sample_rate : int
        Sample rate in Hz
    model_data : dict
        Trained model data
    mf, tr : float
        Masking parameters
    window_sec : float
        Window length in seconds
    increment_sec : float
        Window increment in seconds

    Returns:
    --------
    predictions : array
        Class predictions for each window
    probabilities : array
        Prediction probabilities
    window_starts : array
        Start time in seconds for each window
    """
    classifier = model_data['classifier']
    scaler = model_data['scaler']
    pca = model_data.get('pca', None)

    window_length = int(window_sec * sample_rate)
    increment_shift = int(increment_sec * sample_rate)

    features_list = []
    window_starts = []

    print(f"  Processing windows (window: {window_sec}s, increment: {increment_sec}s)...")
    n_windows = (len(audio) - window_length) // increment_shift + 1

    for i in range(0, len(audio) - window_length, increment_shift):
        if len(window_starts) % 100 == 0:
            print(f"    Window {len(window_starts)+1}/{n_windows}...")

        # Extract window
        audio_window = audio[i:i + window_length]

        # Extract features
        feature_vector = extract_features_from_window(audio_window, mf, tr, sample_rate)
        features_list.append(feature_vector)
        window_starts.append(i / sample_rate)

    features = np.array(features_list)
    window_starts = np.array(window_starts)

    # Preprocess features
    print("  Preprocessing and classifying...")
    features_scaled = scaler.transform(features)

    if pca is not None:
        features_reduced = pca.transform(features_scaled)
    else:
        features_reduced = features_scaled

    # Make predictions
    predictions = classifier.predict(features_reduced)
    probabilities = None
    if hasattr(classifier, 'predict_proba'):
        probabilities = classifier.predict_proba(features_reduced)

    return predictions, probabilities, window_starts


def find_detection_segments(predictions, probabilities, window_starts, window_sec=1.0,
                            target_classes=[0, 1], min_confidence=0.6):
    """
    Find continuous segments with bee or hornet detections.

    Parameters:
    -----------
    predictions : array
        Class predictions
    probabilities : array
        Prediction probabilities
    window_starts : array
        Start times for each window
    window_sec : float
        Window length in seconds
    target_classes : list
        Classes to detect (0=hornet, 1=bee)
    min_confidence : float
        Minimum confidence threshold

    Returns:
    --------
    segments : list of dict
        List of detected segments with metadata
    """
    segments = []

    for i, (pred, start_time) in enumerate(zip(predictions, window_starts)):
        # Check if this window contains bee or hornet
        if pred in target_classes:
            # Check confidence if available
            if probabilities is not None:
                confidence = probabilities[i, pred]
                if confidence < min_confidence:
                    continue
            else:
                confidence = 1.0

            segments.append({
                'start_time': start_time,
                'end_time': start_time + window_sec,
                'class': pred,
                'confidence': confidence
            })

    return segments


def merge_nearby_segments(segments, max_gap=2.0):
    """
    Merge segments that are close together.

    Parameters:
    -----------
    segments : list of dict
        Detection segments
    max_gap : float
        Maximum gap in seconds to merge

    Returns:
    --------
    merged : list of dict
        Merged segments
    """
    if not segments:
        return []

    # Sort by start time
    segments = sorted(segments, key=lambda x: x['start_time'])

    merged = []
    current = segments[0].copy()

    for seg in segments[1:]:
        # Check if this segment is close to current
        if seg['start_time'] - current['end_time'] <= max_gap:
            # Merge
            current['end_time'] = max(current['end_time'], seg['end_time'])
            # Keep highest confidence
            if seg['confidence'] > current['confidence']:
                current['confidence'] = seg['confidence']
                current['class'] = seg['class']
        else:
            # Save current and start new
            merged.append(current)
            current = seg.copy()

    # Add last segment
    merged.append(current)

    return merged


def extract_10s_segments(audio, sample_rate, segments, segment_duration=10.0,
                         overlap_threshold=0.5):
    """
    Extract 10-second segments centered around detections.

    Parameters:
    -----------
    audio : array
        Full audio signal
    sample_rate : int
        Sample rate in Hz
    segments : list of dict
        Detection segments
    segment_duration : float
        Duration of output segments in seconds
    overlap_threshold : float
        Minimum overlap to include a detection in a segment

    Returns:
    --------
    extracted_segments : list of dict
        10-second audio segments with metadata
    """
    extracted = []
    segment_samples = int(segment_duration * sample_rate)

    for seg in segments:
        # Calculate center of detection
        center_time = (seg['start_time'] + seg['end_time']) / 2

        # Calculate 10s window centered on detection
        start_time = max(0, center_time - segment_duration / 2)
        end_time = min(len(audio) / sample_rate, start_time + segment_duration)

        # Adjust if we're at the end
        if end_time - start_time < segment_duration:
            start_time = max(0, end_time - segment_duration)

        # Extract audio
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = audio[start_sample:end_sample]

        # Pad if necessary (at beginning or end of file)
        if len(audio_segment) < segment_samples:
            padding = segment_samples - len(audio_segment)
            if start_sample == 0:
                # Pad at beginning
                audio_segment = np.pad(audio_segment, (padding, 0), mode='constant')
            else:
                # Pad at end
                audio_segment = np.pad(audio_segment, (0, padding), mode='constant')

        extracted.append({
            'audio': audio_segment,
            'start_time': start_time,
            'end_time': end_time,
            'class': seg['class'],
            'confidence': seg['confidence'],
            'detection_center': center_time
        })

    return extracted


def save_segments(segments, output_dir, file_prefix, class_names, sample_rate=44100):
    """
    Save extracted segments as WAV files.

    Parameters:
    -----------
    segments : list of dict
        Extracted audio segments
    output_dir : str
        Output directory
    file_prefix : str
        Prefix for output files
    class_names : list
        Class name mapping
    sample_rate : int
        Sample rate in Hz
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        class_name = class_names[seg['class']].replace(' ', '_')
        confidence = seg['confidence']
        start_time = seg['start_time']

        filename = f"{file_prefix}_{i+1:03d}_{class_name}_conf{confidence:.2f}_t{start_time:.1f}s.wav"
        filepath = os.path.join(output_dir, filename)

        # Save as WAV
        sf.write(filepath, seg['audio'], sample_rate)
        print(f"  Saved: {filename}")


def process_audio_file(audio_file, model_data, mf, tr, output_dir,
                       min_confidence=0.6, segment_duration=10.0):
    """
    Process a single audio file and extract bee/hornet segments.

    Parameters:
    -----------
    audio_file : str
        Path to audio file
    model_data : dict
        Trained model data
    mf, tr : float
        Masking parameters
    output_dir : str
        Output directory
    min_confidence : float
        Minimum confidence threshold
    segment_duration : float
        Duration of output segments
    """
    print(f"\nProcessing: {audio_file}")
    print("=" * 60)

    # Load audio
    audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
    duration = len(audio) / sample_rate
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")

    # Classify windows
    predictions, probabilities, window_starts = classify_audio_windows(
        audio, sample_rate, model_data, mf, tr
    )

    # Count detections
    class_names = model_data['class_names']
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nClassification summary:")
    for class_id, count in zip(unique, counts):
        percentage = 100 * count / len(predictions)
        print(f"  {class_names[class_id]}: {count} windows ({percentage:.1f}%)")

    # Find bee and hornet segments
    print(f"\nFinding bee/hornet segments (min confidence: {min_confidence})...")
    segments = find_detection_segments(
        predictions, probabilities, window_starts,
        target_classes=[0, 1],  # 0=hornet, 1=bee
        min_confidence=min_confidence
    )
    print(f"  Found {len(segments)} detection windows")

    # Merge nearby detections
    merged = merge_nearby_segments(segments, max_gap=2.0)
    print(f"  Merged into {len(merged)} continuous segments")

    if not merged:
        print("  No bee/hornet segments found!")
        return

    # Extract 10-second segments
    print(f"\nExtracting {segment_duration}s segments...")
    extracted = extract_10s_segments(audio, sample_rate, merged, segment_duration)
    print(f"  Extracted {len(extracted)} segments")

    # Save segments
    file_prefix = Path(audio_file).stem.replace(' ', '_')
    print(f"\nSaving segments to {output_dir}/...")
    save_segments(extracted, output_dir, file_prefix, class_names, sample_rate)

    print(f"\n{'='*60}")
    print(f"Completed! Saved {len(extracted)} segments")
    print(f"{'='*60}")


def main():
    """Main function to process shotgun files."""
    print("Bee/Hornet Segment Extractor")
    print("=" * 60)

    # Load model and parameters
    model_data = load_trained_model()
    mf, tr = load_masking_parameters()

    # Find shotgun files
    data_dir = "data"
    shotgun_files = [
        os.path.join(data_dir, "13-10-22 shotgun.wav"),
        os.path.join(data_dir, "13-10-22 shotgun 2.wav")
    ]

    # Check files exist
    existing_files = [f for f in shotgun_files if os.path.exists(f)]
    if not existing_files:
        print(f"Error: No shotgun files found in {data_dir}/")
        return

    print(f"\nFound {len(existing_files)} shotgun files:")
    for f in existing_files:
        print(f"  - {f}")

    # Create output directory
    output_dir = "extracted_segments"

    # Process each file
    for audio_file in existing_files:
        process_audio_file(
            audio_file,
            model_data,
            mf,
            tr,
            output_dir,
            min_confidence=0.6,  # Adjust this to be more/less strict
            segment_duration=10.0
        )

    print("\n" + "=" * 60)
    print("All files processed!")
    print(f"Output saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
