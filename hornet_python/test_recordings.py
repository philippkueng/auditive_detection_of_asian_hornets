"""
Testing and validation script.
Python translation of test_recordings.m

Similar to masking.m but specifically designed for testing the classification
system on validation recordings. Loads specific test data and applies the
polygonal masking classification.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from masking import classify_points, create_polygonal_areas
import argparse


def test_recording(test_file='29-06-23.pkl', shrink_factor=0.1):
    """
    Test classification on a specific recording.

    Parameters:
    -----------
    test_file : str
        Path to the test data pickle file (from CCP analysis)
    shrink_factor : float
        Boundary shrink factor for polygon creation

    Returns:
    --------
    results : dict
        Dictionary containing classifications and statistics
    """
    print(f"Testing recording: {test_file}")
    print("=" * 60)

    # Check if polygonal areas exist, create if not
    if not os.path.exists('polygonal_areas.pkl'):
        print("\nPolygonal areas not found. Creating them from training data...")
        create_polygonal_areas()
        print()

    # Load test data
    print("Loading test data...")
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)

    df_x = test_data['df_x']
    df_y = test_data['df_y']
    index_array = test_data['index_array']

    print(f"  Test points: {len(df_x)}")
    print(f"  Time range: {index_array[0]} to {index_array[-1]} samples")

    # Classify points
    classifications = classify_points(df_x, df_y, shrink_factor)

    # Find indices of each classification
    hornet_indices = np.where(classifications == 999)[0]
    bee_indices = np.where(classifications == 998)[0]
    winter_bg_indices = np.where(classifications == 997)[0]
    summer_bg_indices = np.where(classifications == 996)[0]
    ambiguous_indices = np.where(classifications == 995)[0]

    print("\nDetailed classification results:")
    print(f"  Hornet detections: {len(hornet_indices)}")
    if len(hornet_indices) > 0:
        print(f"    Indices: {hornet_indices[:10]}..." if len(hornet_indices) > 10
              else f"    Indices: {hornet_indices}")

    print(f"  Bee detections: {len(bee_indices)}")
    if len(bee_indices) > 0:
        print(f"    Indices: {bee_indices[:10]}..." if len(bee_indices) > 10
              else f"    Indices: {bee_indices}")

    print(f"  Winter background: {len(winter_bg_indices)}")
    print(f"  Summer background: {len(summer_bg_indices)}")
    print(f"  Ambiguous: {len(ambiguous_indices)}")

    # Create timeline visualization
    print("\nGenerating timeline visualization...")
    fig, ax = plt.subplots(figsize=(16, 4))

    # Convert indices to time (assuming 1 second per window)
    time_points = np.arange(len(classifications))

    # Plot classification over time
    colors = {999: 'red', 998: 'blue', 997: 'black', 996: 'cyan', 995: 'gray'}
    labels = {999: 'Hornet', 998: 'Bee', 997: 'Winter BG',
              996: 'Summer BG', 995: 'Ambiguous'}

    for class_id, color in colors.items():
        mask = classifications == class_id
        if np.any(mask):
            ax.scatter(time_points[mask], np.ones(np.sum(mask)) * class_id,
                      c=color, s=50, alpha=0.7, label=labels[class_id])

    ax.set_xlabel('Time window', fontsize=12)
    ax.set_ylabel('Classification', fontsize=12)
    ax.set_title(f'Classification Timeline: {test_file}', fontsize=14)
    ax.set_yticks([995, 996, 997, 998, 999])
    ax.set_yticklabels(['Ambiguous', 'Summer BG', 'Winter BG', 'Bee', 'Hornet'])
    ax.legend(loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    output_plot = test_file.replace('.pkl', '_timeline.png')
    plt.savefig(output_plot, dpi=150)
    print(f"  Saved {output_plot}")

    # Prepare results
    results = {
        'test_file': test_file,
        'df_x': df_x,
        'df_y': df_y,
        'index_array': index_array,
        'classifications': classifications,
        'hornet_indices': hornet_indices,
        'bee_indices': bee_indices,
        'winter_bg_indices': winter_bg_indices,
        'summer_bg_indices': summer_bg_indices,
        'ambiguous_indices': ambiguous_indices,
        'n_hornet': len(hornet_indices),
        'n_bee': len(bee_indices),
        'n_winter_bg': len(winter_bg_indices),
        'n_summer_bg': len(summer_bg_indices),
        'n_ambiguous': len(ambiguous_indices)
    }

    # Save results
    output_file = test_file.replace('.pkl', '_test_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nTest results saved to {output_file}")
    print("=" * 60)

    return results


def compare_multiple_recordings(test_files):
    """
    Compare classification results across multiple recordings.

    Parameters:
    -----------
    test_files : list of str
        List of test data pickle files
    """
    print("Comparing multiple recordings...")
    print("=" * 60)

    all_results = []
    for test_file in test_files:
        try:
            results = test_recording(test_file)
            all_results.append(results)
        except FileNotFoundError:
            print(f"Warning: File {test_file} not found, skipping...")
            continue

    if len(all_results) == 0:
        print("No valid test files found!")
        return

    # Create comparison visualization
    print("\nGenerating comparison visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))

    categories = ['Hornet', 'Bee', 'Winter BG', 'Summer BG', 'Ambiguous']
    x = np.arange(len(categories))
    width = 0.8 / len(all_results)

    for i, results in enumerate(all_results):
        counts = [
            results['n_hornet'],
            results['n_bee'],
            results['n_winter_bg'],
            results['n_summer_bg'],
            results['n_ambiguous']
        ]
        ax.bar(x + i * width, counts, width, label=results['test_file'])

    ax.set_xlabel('Classification Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Comparison Across Recordings', fontsize=14)
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150)
    print("  Saved comparison_results.png")

    print("=" * 60)


def main():
    """
    Main function for testing recordings.
    """
    parser = argparse.ArgumentParser(
        description='Test classification on audio recordings'
    )
    parser.add_argument('--file', type=str, default='29-06-23.pkl',
                        help='Test data pickle file (from CCP analysis)')
    parser.add_argument('--shrink', type=float, default=0.1,
                        help='Boundary shrink factor (0-1)')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple test files')

    args = parser.parse_args()

    if args.compare:
        # Compare multiple recordings
        compare_multiple_recordings(args.compare)
    else:
        # Test single recording
        test_recording(args.file, args.shrink)


if __name__ == "__main__":
    main()
