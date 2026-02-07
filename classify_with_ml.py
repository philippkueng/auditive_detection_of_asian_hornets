"""
Classify Audio Recordings Using Trained ML Model
Uses the traditional ML classifier trained by train_ml_classifiers.py

This script provides an alternative to the polygon-based classification,
using sklearn's .predict() method directly on the trained model.
"""

import numpy as np
import pickle
import librosa
import matplotlib.pyplot as plt
import argparse
from utils import two_D_FT_Gaussian


def load_trained_model():
    """
    Load the trained ML classifier.

    Returns:
    --------
    model_data : dict
        Dictionary containing classifier, scaler, and metadata
    """
    print("Loading trained ML model...")
    with open('trained_ml_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)

    print(f"  Classifier: {model_data['classifier_name']}")
    if 'n_features_reduced' in model_data:
        print(f"  Features: {model_data['n_features_original']} → {model_data['n_features_reduced']} (PCA)")
    print(f"  CV Accuracy: {model_data['cv_accuracy']:.3f} (+/- {model_data['cv_std']:.3f})")
    print(f"  Training Accuracy: {model_data['train_accuracy']:.3f}")

    return model_data


def extract_features_from_audio(audio_file, start_min=0, start_sec=0,
                                  end_min=0, end_sec=0, increment_sec=1.0):
    """
    Extract 2D FT features from audio file in sliding windows.

    Parameters:
    -----------
    audio_file : str
        Path to audio file
    start_min, start_sec, end_min, end_sec : int/float
        Time range to analyze
    increment_sec : float
        Window increment in seconds

    Returns:
    --------
    features : array, shape (n_windows, n_features)
        Feature vectors for each window
    index_array : array
        Starting sample index for each window
    """
    print(f"Loading audio file: {audio_file}")

    # Load masking parameters for 2D FT computation
    with open('masking_parameters.pkl', 'rb') as f:
        params = pickle.load(f)

    mf = params['mf']
    tr = params['tr']

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
    features_list = []
    index_array = []

    max_time = len(U) / (2 * sample_rate)

    print("\nExtracting features from audio windows...")
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

        # Flatten to feature vector
        feature_vector = scaled_audio.flatten()
        features_list.append(feature_vector)
        index_array.append(start_time)

    features = np.array(features_list)
    index_array = np.array(index_array)

    print(f"\nExtracted features from {len(features)} windows")
    print(f"  Feature vector size: {features.shape[1]}")

    return features, index_array, sample_rate


def classify_audio(audio_file, start_min=0, start_sec=0, end_min=0, end_sec=0,
                   increment_sec=1.0):
    """
    Classify audio file using trained ML model.

    Parameters:
    -----------
    audio_file : str
        Path to audio file
    start_min, start_sec, end_min, end_sec : int/float
        Time range to analyze
    increment_sec : float
        Window increment in seconds

    Returns:
    --------
    results : dict
        Classification results including predictions and probabilities
    """
    # Load trained model
    model_data = load_trained_model()
    classifier = model_data['classifier']
    scaler = model_data['scaler']
    pca = model_data.get('pca', None)  # PCA may not exist in older models
    class_names = model_data['class_names']

    # Extract features
    features, index_array, sample_rate = extract_features_from_audio(
        audio_file, start_min, start_sec, end_min, end_sec, increment_sec
    )

    # Scale and reduce features
    print("\nPreprocessing features...")
    features_scaled = scaler.transform(features)

    if pca is not None:
        print(f"  Applying PCA: {features.shape[1]} → {pca.n_components_} features")
        features_reduced = pca.transform(features_scaled)
    else:
        features_reduced = features_scaled

    # Make predictions
    print("Classifying windows...")
    predictions = classifier.predict(features_reduced)

    # Get prediction probabilities if available
    probabilities = None
    if hasattr(classifier, 'predict_proba'):
        probabilities = classifier.predict_proba(features_reduced)

    # Count classifications
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nClassification results:")
    for class_id, count in zip(unique, counts):
        percentage = 100 * count / len(predictions)
        print(f"  {class_names[class_id]}: {count} ({percentage:.1f}%)")

    results = {
        'predictions': predictions,
        'probabilities': probabilities,
        'index_array': index_array,
        'sample_rate': sample_rate,
        'class_names': class_names,
        'classifier_name': model_data['classifier_name'],
        'audio_file': audio_file
    }

    return results


def visualize_results(results, output_file='ml_classification_result.png'):
    """
    Create visualizations of classification results.

    Parameters:
    -----------
    results : dict
        Classification results
    output_file : str
        Output filename for plot
    """
    predictions = results['predictions']
    probabilities = results['probabilities']
    class_names = results['class_names']
    time_points = np.arange(len(predictions))

    # Create figure with subplots
    if probabilities is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 5))

    # Plot 1: Timeline of classifications
    colors = {0: 'red', 1: 'blue', 2: 'black', 3: 'cyan'}
    labels_shown = set()

    for class_id in range(len(class_names)):
        mask = predictions == class_id
        if np.any(mask):
            label = class_names[class_id] if class_id not in labels_shown else None
            ax1.scatter(time_points[mask], predictions[mask],
                       c=colors[class_id], s=50, alpha=0.7, label=label)
            labels_shown.add(class_id)

    ax1.set_xlabel('Time window', fontsize=12)
    ax1.set_ylabel('Classification', fontsize=12)
    ax1.set_title(f'Classification Timeline - {results["classifier_name"]}', fontsize=14)
    ax1.set_yticks(range(len(class_names)))
    ax1.set_yticklabels(class_names)
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='x', alpha=0.3)

    # Plot 2: Probability heatmap (if available)
    if probabilities is not None:
        im = ax2.imshow(probabilities.T, aspect='auto', cmap='viridis',
                       interpolation='nearest', origin='lower')
        ax2.set_xlabel('Time window', fontsize=12)
        ax2.set_ylabel('Class', fontsize=12)
        ax2.set_title('Classification Probabilities', fontsize=14)
        ax2.set_yticks(range(len(class_names)))
        ax2.set_yticklabels(class_names)
        plt.colorbar(im, ax=ax2, label='Probability')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nVisualization saved to {output_file}")


def save_results(results, output_file):
    """
    Save classification results to pickle file.

    Parameters:
    -----------
    results : dict
        Classification results
    output_file : str
        Output pickle filename
    """
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_file}")


def main():
    """
    Main function for ML-based audio classification.
    """
    parser = argparse.ArgumentParser(
        description='Classify audio recordings using trained ML model'
    )
    parser.add_argument('audio_file', type=str,
                        help='Path to audio file')
    parser.add_argument('--start-min', type=int, default=0,
                        help='Start time in minutes')
    parser.add_argument('--start-sec', type=float, default=0,
                        help='Start time seconds component')
    parser.add_argument('--end-min', type=int, default=0,
                        help='End time in minutes')
    parser.add_argument('--end-sec', type=float, default=0,
                        help='End time seconds component')
    parser.add_argument('--increment', type=float, default=1.0,
                        help='Window increment in seconds')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file name')

    args = parser.parse_args()

    print("=" * 60)
    print("ML-Based Audio Classification")
    print("=" * 60)

    # Classify audio
    results = classify_audio(
        args.audio_file,
        args.start_min,
        args.start_sec,
        args.end_min,
        args.end_sec,
        args.increment
    )

    # Visualize results
    visualize_results(results)

    # Save results
    if args.output:
        output_file = args.output
    else:
        # Generate output filename from input
        import os
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        output_file = f'{base_name}_ml_classified.pkl'

    save_results(results, output_file)

    print("\n" + "=" * 60)
    print("Classification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
