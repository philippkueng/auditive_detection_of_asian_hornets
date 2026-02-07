"""
Compare Traditional Polygon Classification vs ML Classification

This script loads results from both classification approaches and creates
side-by-side visualizations in the discriminant function space for comparison.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Polygon as MplPolygon
from utils import create_polygon_boundary, points_in_polygon


def load_polygon_classification(test_file):
    """
    Load polygon-based classification results.

    Parameters:
    -----------
    test_file : str
        Path to polygon classification results

    Returns:
    --------
    polygon_results : dict
        Polygon classification results
    """
    print(f"Loading polygon classification results from {test_file}...")
    with open(test_file, 'rb') as f:
        results = pickle.load(f)
    return results


def load_ml_classification(test_file):
    """
    Load ML-based classification results.

    Parameters:
    -----------
    test_file : str
        Path to ML classification results

    Returns:
    --------
    ml_results : dict
        ML classification results
    """
    print(f"Loading ML classification results from {test_file}...")
    with open(test_file, 'rb') as f:
        results = pickle.load(f)
    return results


def compute_df_scores(test_file):
    """
    Compute DF scores if they don't exist in the test file.
    This is for ML results that don't have df_x, df_y.

    Parameters:
    -----------
    test_file : str
        Path to CCP results (29-06-23.pkl)

    Returns:
    --------
    df_x, df_y : arrays
        DF scores
    """
    print(f"Loading DF scores from {test_file}...")
    with open(test_file, 'rb') as f:
        data = pickle.load(f)

    if 'df_x' in data and 'df_y' in data:
        return data['df_x'], data['df_y']
    else:
        raise ValueError("Test file does not contain df_x and df_y scores")


def load_polygonal_areas():
    """
    Load or create polygonal area boundaries.

    Returns:
    --------
    poly_data : dict
        Polygonal boundary data
    boundaries : dict
        Computed boundary polygons for each class
    """
    # Load polygonal areas
    with open('polygonal_areas.pkl', 'rb') as f:
        poly_data = pickle.load(f)

    # Create boundaries
    shrink_factor = 0.1

    hornet_points = np.column_stack([poly_data['hornet_X'], poly_data['hornet_Y']])
    bee_points = np.column_stack([poly_data['bee_X'], poly_data['bee_Y']])
    bg_points = np.column_stack([poly_data['bg_X'], poly_data['bg_Y']])
    bgs_points = np.column_stack([poly_data['bgs_X'], poly_data['bgs_Y']])

    boundaries = {
        'hornet': create_polygon_boundary(hornet_points, shrink_factor),
        'bee': create_polygon_boundary(bee_points, shrink_factor),
        'winter_bg': create_polygon_boundary(bg_points, shrink_factor),
        'summer_bg': create_polygon_boundary(bgs_points, shrink_factor)
    }

    return poly_data, boundaries


def map_ml_to_polygon_codes(ml_predictions):
    """
    Map ML class labels (0,1,2,3) to polygon codes (999,998,997,996).

    Parameters:
    -----------
    ml_predictions : array
        ML predictions (0=hornet, 1=bee, 2=winter, 3=summer)

    Returns:
    --------
    polygon_codes : array
        Mapped to polygon codes
    """
    mapping = {0: 999, 1: 998, 2: 997, 3: 996}
    return np.array([mapping[p] for p in ml_predictions])


def create_comparison_visualization(df_x, df_y, polygon_classifications, ml_classifications,
                                     poly_data, boundaries, output_file='classification_comparison.png'):
    """
    Create side-by-side comparison visualization.

    Parameters:
    -----------
    df_x, df_y : arrays
        DF scores for test points
    polygon_classifications : array
        Polygon-based classifications
    ml_classifications : array
        ML-based classifications
    poly_data : dict
        Polygonal area training data
    boundaries : dict
        Polygon boundaries
    output_file : str
        Output filename
    """
    print("\nGenerating comparison visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Color mapping
    colors = {999: 'red', 998: 'blue', 997: 'black', 996: 'cyan', 995: 'gray'}
    labels = {999: 'Hornet', 998: 'Bee', 997: 'Winter BG', 996: 'Summer BG', 995: 'Ambiguous'}

    # Plot 1: Polygon-based classification
    ax1.set_title('Polygon-Based Classification', fontsize=16, fontweight='bold')

    # Draw polygons
    if len(boundaries['hornet']) > 2:
        polygon = MplPolygon(boundaries['hornet'], alpha=0.2, facecolor='red',
                             edgecolor='red', linewidth=2)
        ax1.add_patch(polygon)

    if len(boundaries['bee']) > 2:
        polygon = MplPolygon(boundaries['bee'], alpha=0.2, facecolor='blue',
                             edgecolor='blue', linewidth=2)
        ax1.add_patch(polygon)

    if len(boundaries['winter_bg']) > 2:
        polygon = MplPolygon(boundaries['winter_bg'], alpha=0.2, facecolor='black',
                             edgecolor='black', linewidth=2)
        ax1.add_patch(polygon)

    if len(boundaries['summer_bg']) > 2:
        polygon = MplPolygon(boundaries['summer_bg'], alpha=0.2, facecolor='cyan',
                             edgecolor='cyan', linewidth=2)
        ax1.add_patch(polygon)

    # Plot training data points (small)
    ax1.plot(poly_data['hornet_X'], poly_data['hornet_Y'], 'ro', markersize=4, alpha=0.3)
    ax1.plot(poly_data['bee_X'], poly_data['bee_Y'], 'bo', markersize=4, alpha=0.3)
    ax1.plot(poly_data['bg_X'], poly_data['bg_Y'], 'ko', markersize=4, alpha=0.3)
    ax1.plot(poly_data['bgs_X'], poly_data['bgs_Y'], 'co', markersize=4, alpha=0.3)

    # Plot test points colored by polygon classification
    for class_code, color in colors.items():
        mask = polygon_classifications == class_code
        if np.any(mask):
            ax1.scatter(df_x[mask], df_y[mask], c=color, s=100, alpha=0.8,
                       edgecolors='white', linewidths=1.5, label=labels[class_code],
                       marker='o', zorder=5)

    ax1.set_xlabel('DF Score 1', fontsize=14)
    ax1.set_ylabel('DF Score 2', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: ML-based classification
    ax2.set_title('ML-Based Classification', fontsize=16, fontweight='bold')

    # Draw same polygons for reference
    if len(boundaries['hornet']) > 2:
        polygon = MplPolygon(boundaries['hornet'], alpha=0.1, facecolor='red',
                             edgecolor='red', linewidth=1, linestyle='--')
        ax2.add_patch(polygon)

    if len(boundaries['bee']) > 2:
        polygon = MplPolygon(boundaries['bee'], alpha=0.1, facecolor='blue',
                             edgecolor='blue', linewidth=1, linestyle='--')
        ax2.add_patch(polygon)

    if len(boundaries['winter_bg']) > 2:
        polygon = MplPolygon(boundaries['winter_bg'], alpha=0.1, facecolor='black',
                             edgecolor='black', linewidth=1, linestyle='--')
        ax2.add_patch(polygon)

    if len(boundaries['summer_bg']) > 2:
        polygon = MplPolygon(boundaries['summer_bg'], alpha=0.1, facecolor='cyan',
                             edgecolor='cyan', linewidth=1, linestyle='--')
        ax2.add_patch(polygon)

    # Plot training data points (small, faded)
    ax2.plot(poly_data['hornet_X'], poly_data['hornet_Y'], 'ro', markersize=4, alpha=0.2)
    ax2.plot(poly_data['bee_X'], poly_data['bee_Y'], 'bo', markersize=4, alpha=0.2)
    ax2.plot(poly_data['bg_X'], poly_data['bg_Y'], 'ko', markersize=4, alpha=0.2)
    ax2.plot(poly_data['bgs_X'], poly_data['bgs_Y'], 'co', markersize=4, alpha=0.2)

    # Plot test points colored by ML classification
    ml_codes = map_ml_to_polygon_codes(ml_classifications)
    for class_code, color in colors.items():
        if class_code == 995:  # ML doesn't have ambiguous category
            continue
        mask = ml_codes == class_code
        if np.any(mask):
            ax2.scatter(df_x[mask], df_y[mask], c=color, s=100, alpha=0.8,
                       edgecolors='white', linewidths=1.5, label=labels[class_code],
                       marker='s', zorder=5)  # squares for ML

    ax2.set_xlabel('DF Score 1', fontsize=14)
    ax2.set_ylabel('DF Score 2', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, 'Polygons shown as dashed reference',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved {output_file}")


def create_agreement_analysis(polygon_classifications, ml_classifications):
    """
    Analyze agreement between polygon and ML classifications.

    Parameters:
    -----------
    polygon_classifications : array
        Polygon-based classifications
    ml_classifications : array
        ML-based classifications
    """
    print("\n" + "=" * 60)
    print("Classification Agreement Analysis")
    print("=" * 60)

    # Map ML codes to match polygon codes
    ml_codes = map_ml_to_polygon_codes(ml_classifications)

    # Filter out ambiguous polygon classifications
    valid_mask = polygon_classifications != 995
    polygon_valid = polygon_classifications[valid_mask]
    ml_valid = ml_codes[valid_mask]

    # Calculate agreement
    agreement = np.sum(polygon_valid == ml_valid)
    total_valid = len(polygon_valid)
    agreement_pct = 100 * agreement / total_valid if total_valid > 0 else 0

    print(f"\nTotal test points: {len(polygon_classifications)}")
    print(f"Polygon ambiguous classifications: {np.sum(~valid_mask)}")
    print(f"Valid classifications: {total_valid}")
    print(f"Agreement: {agreement}/{total_valid} ({agreement_pct:.1f}%)")
    print(f"Disagreement: {total_valid - agreement}/{total_valid} ({100 - agreement_pct:.1f}%)")

    # Class-by-class comparison
    print("\nClass-by-class counts:")
    print(f"{'Class':<15} {'Polygon':<10} {'ML':<10}")
    print("-" * 35)

    for code, label in [(999, 'Hornet'), (998, 'Bee'),
                        (997, 'Winter BG'), (996, 'Summer BG')]:
        poly_count = np.sum(polygon_classifications == code)
        ml_count = np.sum(ml_codes == code)
        print(f"{label:<15} {poly_count:<10} {ml_count:<10}")

    ambig_count = np.sum(polygon_classifications == 995)
    print(f"{'Ambiguous':<15} {ambig_count:<10} {'-':<10}")

    # Disagreement analysis
    if total_valid > 0 and agreement < total_valid:
        print("\nDisagreement details:")
        disagreement_mask = valid_mask & (polygon_classifications != ml_codes)
        disagreements = np.column_stack([
            polygon_classifications[disagreement_mask],
            ml_codes[disagreement_mask]
        ])

        code_to_label = {999: 'Hornet', 998: 'Bee', 997: 'Winter', 996: 'Summer'}
        print(f"{'Polygon':<10} -> {'ML':<10} {'Count':<10}")
        print("-" * 30)

        unique_pairs, counts = np.unique(disagreements, axis=0, return_counts=True)
        for pair, count in zip(unique_pairs, counts):
            poly_label = code_to_label.get(pair[0], str(pair[0]))
            ml_label = code_to_label.get(pair[1], str(pair[1]))
            print(f"{poly_label:<10} -> {ml_label:<10} {count:<10}")


def main():
    """
    Main function for comparing classifications.
    """
    parser = argparse.ArgumentParser(
        description='Compare polygon and ML classification results'
    )
    parser.add_argument('--ccp-file', type=str, default='29-06-23.pkl',
                        help='CCP results file with DF scores')
    parser.add_argument('--polygon-file', type=str, default='29-06-23_test_results.pkl',
                        help='Polygon classification results')
    parser.add_argument('--ml-file', type=str, default='29-06-23_ml_classified.pkl',
                        help='ML classification results')
    parser.add_argument('--output', type=str, default='classification_comparison.png',
                        help='Output comparison plot filename')

    args = parser.parse_args()

    print("=" * 60)
    print("Classification Comparison Tool")
    print("=" * 60)

    # Load DF scores from CCP file
    df_x, df_y = compute_df_scores(args.ccp_file)
    print(f"  Loaded {len(df_x)} test points")

    # Load polygon classification
    try:
        polygon_results = load_polygon_classification(args.polygon_file)
        polygon_classifications = polygon_results['classifications']
    except FileNotFoundError:
        print(f"\nError: Polygon classification file '{args.polygon_file}' not found.")
        print("Please run: python test_recordings.py --file 29-06-23.pkl")
        return

    # Load ML classification
    try:
        ml_results = load_ml_classification(args.ml_file)
        ml_classifications = ml_results['predictions']
    except FileNotFoundError:
        print(f"\nError: ML classification file '{args.ml_file}' not found.")
        print("Please run: python classify_with_ml.py data/29-06-23.wav ...")
        return

    # Verify same number of points
    if len(polygon_classifications) != len(ml_classifications):
        print(f"\nWarning: Different number of classifications!")
        print(f"  Polygon: {len(polygon_classifications)}")
        print(f"  ML: {len(ml_classifications)}")
        min_len = min(len(polygon_classifications), len(ml_classifications))
        polygon_classifications = polygon_classifications[:min_len]
        ml_classifications = ml_classifications[:min_len]
        df_x = df_x[:min_len]
        df_y = df_y[:min_len]
        print(f"  Using first {min_len} points for comparison")

    # Load polygonal areas
    poly_data, boundaries = load_polygonal_areas()

    # Create comparison visualization
    create_comparison_visualization(
        df_x, df_y,
        polygon_classifications,
        ml_classifications,
        poly_data,
        boundaries,
        args.output
    )

    # Analyze agreement
    create_agreement_analysis(polygon_classifications, ml_classifications)

    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
