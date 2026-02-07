"""
Export trained polygon-based model to C++ header files for ESP32

This script exports:
1. Discriminant function spectra (new_dfa, new_dfa2)
2. Polygon boundary coordinates for each class
3. Processing parameters (mf, tr, sample_rate, etc.)
"""

import numpy as np
import pickle
import os


def array_to_cpp(arr, name, dtype='float'):
    """Convert numpy array to C++ array declaration."""
    if len(arr.shape) == 1:
        # 1D array
        values = ', '.join([f'{v:.6f}f' for v in arr])
        return f'const {dtype} {name}[{arr.shape[0]}] PROGMEM = {{{values}}};'
    elif len(arr.shape) == 2:
        # 2D array - flatten and store dimensions
        flat = arr.flatten()
        values = ', '.join([f'{v:.6f}f' for v in flat])
        return f'''// Shape: [{arr.shape[0]}, {arr.shape[1]}]
const uint16_t {name}_rows = {arr.shape[0]};
const uint16_t {name}_cols = {arr.shape[1]};
const {dtype} {name}[{arr.shape[0] * arr.shape[1]}] PROGMEM = {{
    {values}
}};'''
    else:
        raise ValueError(f"Cannot convert {len(arr.shape)}D array")


def export_model_to_cpp(output_dir='../hornet_esp32/include'):
    """
    Export trained model to C++ header file.

    Parameters:
    -----------
    output_dir : str
        Directory to save the generated header file
    """
    print("=" * 60)
    print("Exporting Model to C++ Header")
    print("=" * 60)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load masking parameters
    print("\nLoading masking parameters...")
    with open('masking_parameters.pkl', 'rb') as f:
        params = pickle.load(f)

    new_dfa = params['new_dfa']
    new_dfa2 = params['new_dfa2']
    mf = params['mf']
    tr = params['tr']

    print(f"  Discriminant spectrum 1 shape: {new_dfa.shape}")
    print(f"  Discriminant spectrum 2 shape: {new_dfa2.shape}")
    print(f"  mf: {mf} Hz")
    print(f"  tr: {tr} s")

    # Load polygonal areas
    print("\nLoading polygonal areas...")
    with open('polygonal_areas.pkl', 'rb') as f:
        poly_data = pickle.load(f)

    hornet_X = poly_data['hornet_X']
    hornet_Y = poly_data['hornet_Y']
    bee_X = poly_data['bee_X']
    bee_Y = poly_data['bee_Y']
    bg_X = poly_data['bg_X']
    bg_Y = poly_data['bg_Y']
    bgs_X = poly_data['bgs_X']
    bgs_Y = poly_data['bgs_Y']

    print(f"  Hornet boundary points: {len(hornet_X)}")
    print(f"  Bee boundary points: {len(bee_X)}")
    print(f"  Winter BG boundary points: {len(bg_X)}")
    print(f"  Summer BG boundary points: {len(bgs_X)}")

    # Create polygon boundaries (simplified convex hull)
    from scipy.spatial import ConvexHull

    def get_hull_points(x, y):
        points = np.column_stack([x, y])
        if len(points) < 3:
            return x, y
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            return hull_points[:, 0], hull_points[:, 1]
        except:
            return x, y

    hornet_hull_x, hornet_hull_y = get_hull_points(hornet_X, hornet_Y)
    bee_hull_x, bee_hull_y = get_hull_points(bee_X, bee_Y)
    bg_hull_x, bg_hull_y = get_hull_points(bg_X, bg_Y)
    bgs_hull_x, bgs_hull_y = get_hull_points(bgs_X, bgs_Y)

    print(f"\nAfter convex hull:")
    print(f"  Hornet: {len(hornet_hull_x)} vertices")
    print(f"  Bee: {len(bee_hull_x)} vertices")
    print(f"  Winter BG: {len(bg_hull_x)} vertices")
    print(f"  Summer BG: {len(bgs_hull_x)} vertices")

    # Generate C++ header file
    print(f"\nGenerating C++ header file...")
    output_file = os.path.join(output_dir, 'model_data.h')

    with open(output_file, 'w') as f:
        f.write('''/**
 * Hornet Detection Model Data
 * Auto-generated from Python training - DO NOT EDIT MANUALLY
 *
 * This file contains:
 * - Discriminant function spectra for computing DF scores
 * - Polygon boundaries for classification
 * - Processing parameters
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <Arduino.h>

// Processing parameters
const float MODEL_MF = ''' + f'{mf:.6f}f' + ''';  // Spectral repetition (Hz)
const float MODEL_TR = ''' + f'{tr:.6f}f' + ''';  // Time resolution (s)
const uint16_t MODEL_FREQ_MIN = 4;      // Minimum frequency index
const uint16_t MODEL_FREQ_MAX = 60;     // Maximum frequency index
const uint16_t MODEL_SAMPLE_RATE = 48000;  // Expected sample rate (Hz)
const float MODEL_WINDOW_LENGTH = 1.0f;    // Window length (s)

// Discriminant Function Spectra
// These are used to compute DF scores from 2D FT
''')

        # Write discriminant spectra
        f.write(array_to_cpp(new_dfa, 'DF_SPECTRUM_1') + '\n\n')
        f.write(array_to_cpp(new_dfa2, 'DF_SPECTRUM_2') + '\n\n')

        # Write polygon boundaries
        f.write('''
// Classification Polygon Boundaries
// Format: arrays of X and Y coordinates for each class polygon

''')

        # Hornet polygon
        f.write(f'const uint8_t POLYGON_HORNET_SIZE = {len(hornet_hull_x)};\n')
        f.write(array_to_cpp(hornet_hull_x, 'POLYGON_HORNET_X') + '\n')
        f.write(array_to_cpp(hornet_hull_y, 'POLYGON_HORNET_Y') + '\n\n')

        # Bee polygon
        f.write(f'const uint8_t POLYGON_BEE_SIZE = {len(bee_hull_x)};\n')
        f.write(array_to_cpp(bee_hull_x, 'POLYGON_BEE_X') + '\n')
        f.write(array_to_cpp(bee_hull_y, 'POLYGON_BEE_Y') + '\n\n')

        # Winter BG polygon
        f.write(f'const uint8_t POLYGON_WINTER_SIZE = {len(bg_hull_x)};\n')
        f.write(array_to_cpp(bg_hull_x, 'POLYGON_WINTER_X') + '\n')
        f.write(array_to_cpp(bg_hull_y, 'POLYGON_WINTER_Y') + '\n\n')

        # Summer BG polygon
        f.write(f'const uint8_t POLYGON_SUMMER_SIZE = {len(bgs_hull_x)};\n')
        f.write(array_to_cpp(bgs_hull_x, 'POLYGON_SUMMER_X') + '\n')
        f.write(array_to_cpp(bgs_hull_y, 'POLYGON_SUMMER_Y') + '\n\n')

        # Classification codes
        f.write('''
// Classification result codes
#define CLASS_HORNET 999
#define CLASS_BEE 998
#define CLASS_WINTER_BG 997
#define CLASS_SUMMER_BG 996
#define CLASS_AMBIGUOUS 995

#endif // MODEL_DATA_H
''')

    print(f"  Saved to: {output_file}")

    # Calculate approximate memory usage
    spectrum_size = new_dfa.size * 4 + new_dfa2.size * 4  # 4 bytes per float
    polygon_size = (len(hornet_hull_x) + len(hornet_hull_y) +
                    len(bee_hull_x) + len(bee_hull_y) +
                    len(bg_hull_x) + len(bg_hull_y) +
                    len(bgs_hull_x) + len(bgs_hull_y)) * 4
    total_size = spectrum_size + polygon_size

    print(f"\nMemory usage estimate:")
    print(f"  Discriminant spectra: {spectrum_size / 1024:.2f} KB")
    print(f"  Polygon boundaries: {polygon_size / 1024:.2f} KB")
    print(f"  Total: {total_size / 1024:.2f} KB")

    if total_size > 100 * 1024:
        print(f"\n  WARNING: Model data is quite large!")
        print(f"  Consider reducing polygon resolution or using external storage.")

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Copy {output_file} to your ESP32 project")
    print(f"2. Include it in your main.cpp: #include \"model_data.h\"")
    print(f"3. Build and flash your ESP32 project")


if __name__ == "__main__":
    export_model_to_cpp()
