"""
Quick test of the optimized two_D_FT_Gaussian function
"""
import numpy as np
import pickle
import time
from utils import two_D_FT_Gaussian

print("Loading sound files...")
with open('sound_files.pkl', 'rb') as f:
    sound_files = pickle.load(f)

A = sound_files['A']
sample_rate = sound_files['sample_rate']

# Test parameters
window_length = 0.5
mf = 4
tr = 0.04
max_time = len(A) / (2 * sample_rate)

# Extract a small test window
test_timing = 21  # seconds
lower_limit = int((test_timing - window_length) * sample_rate)
upper_limit = int((test_timing + window_length) * sample_rate)
test_window = A[lower_limit:upper_limit]

print(f"\nTest window shape: {test_window.shape}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Window length: {len(test_window) / sample_rate:.3f} seconds")

print("\nComputing 2D FT...")
start_time = time.time()
tdft = two_D_FT_Gaussian(test_window, mf, tr, sample_rate, max_time)
elapsed = time.time() - start_time

print(f"\n2D FT computed in {elapsed:.2f} seconds")
print(f"Output shape: {tdft.shape}")
print(f"Output range: [{tdft.min():.6f}, {tdft.max():.6f}]")

# Check if we can crop to [4:60, :]
if tdft.shape[0] >= 60:
    cropped = tdft[4:60, :]
    print(f"\nCropped shape [4:60, :]: {cropped.shape}")
    print("✓ Dimensions are compatible!")
else:
    print(f"\n✗ Warning: Output has only {tdft.shape[0]} frequency bins, cannot crop to [4:60]")
    print("  You may need to adjust the frequency resolution")
