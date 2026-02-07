"""
Classification using polygonal boundary masks.
Python translation of masking.m

Creates polygonal boundaries around each class in the discriminant function space
and classifies new data points based on which polygon they fall into.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import create_polygon_boundary, points_in_polygon


def create_polygonal_areas():
    """
    Create polygonal areas from training data masking parameters.

    Returns:
    --------
    polygons : dict
        Dictionary containing boundary polygons for each class
    """
    print("Loading masking parameters...")
    with open('masking_parameters.pkl', 'rb') as f:
        params = pickle.load(f)

    A_x = params['A_x']
    A_y = params['A_y']
    hornet_wavs = params['hornet_wavs']
    bee_wavs = params['bee_wavs']
    bg_wavs = params['bg_wavs']
    bgs_wavs = params['bgs_wavs']

    # Separate by category
    hornet_X = []
    bee_X = []
    bg_X = []
    bgs_X = []

    for pulse in range(len(A_x)):
        if pulse < hornet_wavs:
            hornet_X.append(A_x[pulse])
        elif pulse < hornet_wavs + bee_wavs:
            bee_X.append(A_x[pulse])
        elif pulse < hornet_wavs + bee_wavs + bg_wavs:
            bg_X.append(A_x[pulse])
        else:
            bgs_X.append(A_x[pulse])

    hornet_Y = []
    bee_Y = []
    bg_Y = []
    bgs_Y = []

    for pulse in range(len(A_y)):
        if pulse < hornet_wavs:
            hornet_Y.append(A_y[pulse])
        elif pulse < hornet_wavs + bee_wavs:
            bee_Y.append(A_y[pulse])
        elif pulse < hornet_wavs + bee_wavs + bg_wavs:
            bg_Y.append(A_y[pulse])
        else:
            bgs_Y.append(A_y[pulse])

    # Convert to arrays
    hornet_X = np.array(hornet_X)
    hornet_Y = np.array(hornet_Y)
    bee_X = np.array(bee_X)
    bee_Y = np.array(bee_Y)
    bg_X = np.array(bg_X)
    bg_Y = np.array(bg_Y)
    bgs_X = np.array(bgs_X)
    bgs_Y = np.array(bgs_Y)

    # Visualize training data
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(bee_X, bee_Y, 'bo', label='Bee')
    ax.plot(hornet_X, hornet_Y, 'ro', label='Hornet')
    ax.plot(bg_X, bg_Y, 'ko', label='Winter background')
    ax.plot(bgs_X, bgs_Y, 'co', label='Summer background')
    ax.set_xlabel('DF score 1')
    ax.set_ylabel('DF score 2')
    ax.set_title('Training Data Points')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('polygonal_areas.png', dpi=150)
    print("  Saved polygonal_areas.png")

    # Save polygonal areas
    polygonal_data = {
        'bee_X': bee_X,
        'bee_Y': bee_Y,
        'hornet_X': hornet_X,
        'hornet_Y': hornet_Y,
        'bg_X': bg_X,
        'bg_Y': bg_Y,
        'bgs_X': bgs_X,
        'bgs_Y': bgs_Y
    }

    with open('polygonal_areas.pkl', 'wb') as f:
        pickle.dump(polygonal_data, f)

    print("  Saved polygonal_areas.pkl")

    return polygonal_data


def classify_points(df_x, df_y, shrink_factor=0.1):
    """
    Classify test points using polygonal boundary masks.

    Parameters:
    -----------
    df_x : array
        DF score 1 for test points
    df_y : array
        DF score 2 for test points
    shrink_factor : float
        Boundary shrink factor (0-1)

    Returns:
    --------
    classifications : array
        Classification for each point:
        999 = Hornet, 998 = Bee, 997 = Winter background,
        996 = Summer background, 995 = No category
    """
    print("\nLoading polygonal areas...")
    with open('polygonal_areas.pkl', 'rb') as f:
        poly_data = pickle.load(f)

    # Create boundary polygons for each class
    print("Creating boundary polygons...")
    hornet_points = np.column_stack([poly_data['hornet_X'], poly_data['hornet_Y']])
    bee_points = np.column_stack([poly_data['bee_X'], poly_data['bee_Y']])
    bg_points = np.column_stack([poly_data['bg_X'], poly_data['bg_Y']])
    bgs_points = np.column_stack([poly_data['bgs_X'], poly_data['bgs_Y']])

    hornet_boundary = create_polygon_boundary(hornet_points, shrink_factor)
    bee_boundary = create_polygon_boundary(bee_points, shrink_factor)
    bg_boundary = create_polygon_boundary(bg_points, shrink_factor)
    bgs_boundary = create_polygon_boundary(bgs_points, shrink_factor)

    # Visualize boundaries and test points
    print("\nGenerating classification visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot boundaries
    if len(hornet_boundary) > 2:
        polygon = plt.Polygon(hornet_boundary, alpha=0.3, facecolor='red',
                              edgecolor='red', linewidth=2, label='Hornet region')
        ax.add_patch(polygon)
        ax.plot(poly_data['hornet_X'], poly_data['hornet_Y'], 'ro',
                markeredgecolor='black', markersize=6)

    if len(bee_boundary) > 2:
        polygon = plt.Polygon(bee_boundary, alpha=0.3, facecolor='blue',
                              edgecolor='blue', linewidth=2, label='Bee region')
        ax.add_patch(polygon)
        ax.plot(poly_data['bee_X'], poly_data['bee_Y'], 'bo',
                markeredgecolor='white', markersize=6)

    if len(bg_boundary) > 2:
        polygon = plt.Polygon(bg_boundary, alpha=0.3, facecolor='black',
                              edgecolor='black', linewidth=2, label='Winter BG region')
        ax.add_patch(polygon)
        ax.plot(poly_data['bg_X'], poly_data['bg_Y'], 'ko',
                markeredgecolor='white', markersize=6)

    if len(bgs_boundary) > 2:
        polygon = plt.Polygon(bgs_boundary, alpha=0.3, facecolor='cyan',
                              edgecolor='cyan', linewidth=2, label='Summer BG region')
        ax.add_patch(polygon)
        ax.plot(poly_data['bgs_X'], poly_data['bgs_Y'], 'co',
                markeredgecolor='black', markersize=6)

    # Plot test points
    ax.plot(df_x, df_y, 'go', markeredgecolor='green', markersize=8,
            markerfacecolor='none', linewidth=2, label='Test points')

    ax.set_xlabel('DF score 1', fontsize=14)
    ax.set_ylabel('DF score 2', fontsize=14)
    ax.set_title('Classification Boundaries and Test Points', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('classification_result.png', dpi=150)
    print("  Saved classification_result.png")

    # Classify each test point
    print("\nClassifying test points...")
    test_points = np.column_stack([df_x, df_y])
    classifications = np.full(len(df_x), 995)  # Default: no category

    for i, point in enumerate(test_points):
        point_2d = point.reshape(1, -1)

        # Check which polygon the point falls into
        in_hornet = points_in_polygon(point_2d, hornet_boundary)[0] if len(hornet_boundary) > 2 else False
        in_bee = points_in_polygon(point_2d, bee_boundary)[0] if len(bee_boundary) > 2 else False
        in_bg = points_in_polygon(point_2d, bg_boundary)[0] if len(bg_boundary) > 2 else False
        in_bgs = points_in_polygon(point_2d, bgs_boundary)[0] if len(bgs_boundary) > 2 else False

        # Classify based on exclusive membership
        if in_hornet and not in_bee and not in_bg and not in_bgs:
            classifications[i] = 999  # Hornet
        elif not in_hornet and in_bee and not in_bg and not in_bgs:
            classifications[i] = 998  # Bee
        elif not in_hornet and not in_bee and in_bg and not in_bgs:
            classifications[i] = 997  # Winter background
        elif not in_hornet and not in_bee and not in_bg and in_bgs:
            classifications[i] = 996  # Summer background
        else:
            classifications[i] = 995  # No category (ambiguous)

    # Count classifications
    n_hornet = np.sum(classifications == 999)
    n_bee = np.sum(classifications == 998)
    n_winter_bg = np.sum(classifications == 997)
    n_summer_bg = np.sum(classifications == 996)
    n_ambiguous = np.sum(classifications == 995)

    print(f"\nClassification results:")
    print(f"  Hornet: {n_hornet} ({100*n_hornet/len(df_x):.1f}%)")
    print(f"  Bee: {n_bee} ({100*n_bee/len(df_x):.1f}%)")
    print(f"  Winter background: {n_winter_bg} ({100*n_winter_bg/len(df_x):.1f}%)")
    print(f"  Summer background: {n_summer_bg} ({100*n_summer_bg/len(df_x):.1f}%)")
    print(f"  Ambiguous: {n_ambiguous} ({100*n_ambiguous/len(df_x):.1f}%)")

    return classifications


def main():
    """
    Main function to perform masking classification.
    """
    print("Starting masking classification...")
    print("=" * 60)

    # Create polygonal areas from training data
    create_polygonal_areas()

    # Load test data (from CCP analysis)
    print("\nLoading test data...")
    with open('29-06-23.pkl', 'rb') as f:
        test_data = pickle.load(f)

    df_x = test_data['df_x']
    df_y = test_data['df_y']
    index_array = test_data['index_array']

    print(f"  Test points: {len(df_x)}")

    # Classify points
    classifications = classify_points(df_x, df_y, shrink_factor=0.1)

    # Save results
    results = {
        'df_x': df_x,
        'df_y': df_y,
        'index_array': index_array,
        'classifications': classifications
    }

    output_file = '29-06-23_classified.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nClassification results saved to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
