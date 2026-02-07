# Hornet Detection and Classification System (Python)

An llm-assisted Python translation of the MATLAB-based acoustic detection and classification system for Asian hornets (*Vespa velutina*), distinguishing them from honeybees and background noise.

## Original Research

- **Code**: https://github.com/HThomasntu/hornet_TDB_MLA_matlab
- **Paper**: https://www.sciencedirect.com/science/article/pii/S0168169925004132#b0075

## Overview

This system uses a pipeline based on 2D Fourier Transform analysis, Principal Component Analysis (PCA), and Discriminant Function Analysis (DFA) to classify audio segments into four categories:

- **Hornets** (Asian hornets, *Vespa velutina*)
- **Bees** (Honeybees)
- **Winter background noise**
- **Summer background noise**

![Reproduced classification result](classification_result.png)

## ESP32 Embedded Implementation

**NEW**: This project now includes a complete ESP32 implementation for real-time hornet detection on embedded hardware!

- **Location**: `../hornet_esp32/`
- **Features**:
  - Real-time audio processing with I2S microphone
  - Polygon-based classification on-device
  - Serial console alerts for hornet detections
  - ~4-6 second detection latency
- **Hardware**: ESP32 + INMP441 microphone (~$10 total)
- **Quick Start**: See `hornet_esp32/QUICKSTART.md`

This allows deployment of the trained detection model in the field without requiring a computer!

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create the data directory structure:
```bash
mkdir -p data
```

4. Place your audio files in `data/`

```
13-10-22 shotgun 2.wav
13-10-22 shotgun.wav
15-12-22 B_with_sound_02.wav
15-12-22_with_sound_02.wav
16-05-23.wav
27-07-23.wav
29-06-23.wav
```

Download them via the links here: https://www.sciencedirect.com/science/article/pii/S0168169925004132#s0075 (Data availability statement)

## File Descriptions

### Core Scripts

1. **`utils.py`**
   - Utility functions for signal processing
   - `two_D_FT_Gaussian()`: Computes 2D Fourier Transform with Gaussian window
   - `dfa()`: Discriminant Function Analysis using Linear Discriminant Analysis
   - `PCA_deviations()`: Plots PCA decay curve
   - `create_polygon_boundary()`: Creates classification boundaries
   - `points_in_polygon()`: Tests if points fall within boundaries

2. **`read_sound_files.py`**
   - Loads and preprocesses WAV audio files
   - Extracts specific time segments for analysis
   - Saves to `sound_files.pkl`

3. **`TDB.py`** (Training Database)
   - Creates training database from labeled audio samples
   - Computes 2D Fourier Transforms for each category
   - Normalizes spectra by maximum values
   - Saves to `fourth_TDB.pkl`

4. **`PCA_DFA.py`**
   - Performs PCA for dimensionality reduction
   - Applies DFA with 9 principal components
   - Creates two discriminant functions
   - Generates discriminant spectra and centroids
   - Saves to `masking_parameters.pkl`

5. **`CCP.py`** (Continuous Cross-correlation Product)
   - Analyzes test recordings in sliding windows
   - Computes discriminant function scores for each window
   - Saves DF scores and time indices

6. **`masking.py`**
   - Creates polygonal boundaries around each class
   - Classifies test points using `inpolygon` logic
   - Assigns categories: 999=Hornet, 998=Bee, 997=Winter BG, 996=Summer BG, 995=Ambiguous

7. **`test_recordings.py`**
   - Tests classification on validation recordings
   - Generates timeline visualizations
   - Compares multiple recordings

8. **`train_ml_classifiers.py`** (NEW - Traditional ML Approach)
   - Alternative to geometric polygon classification
   - Trains multiple ML classifiers: LDA, SVM, Random Forest, Gradient Boosting, Neural Network
   - Compares performance using cross-validation
   - Selects and saves best model
   - Creates performance visualizations

9. **`classify_with_ml.py`** (NEW - ML-Based Classification)
   - Uses trained ML model for direct classification
   - Alternative to polygon-based masking approach
   - Provides probability estimates for predictions
   - Creates timeline and probability visualizations

10. **`compare_classifications.py`** (NEW - Comparison Tool)
   - Compares polygon vs ML classification results side-by-side
   - Visualizes both approaches in discriminant function space
   - Analyzes agreement/disagreement between methods
   - Creates detailed comparison statistics

## Workflows

There are two classification approaches available:

### Approach A: Geometric Polygon Classification (Original)

This is the original research approach using template matching + geometric boundaries.

### Common Steps (Both Approaches)

#### 1. Prepare Audio Data

Update file paths in `read_sound_files.py` to point to your audio files:

```python
base_dir = "data/"
```

Run the script:
```bash
python read_sound_files.py
```

**Output**: `sound_files.pkl`

### 2. Create Training Database

Ensure timing arrays in `TDB.py` match your labeled training data:

```python
timings_IHND = np.array([21, 67, 75, 81, ...])  # Irregular hornet, not detected
timings_IHD = np.array([15, 37, 63, 85, ...])   # Irregular hornet, detected
# ... etc
```

Run the script:
```bash
python TDB.py
```

**Outputs**:
- `fourth_TDB.pkl`: Training database with scaled 2D Fourier Transforms
- `example_2dft.png`: Visualization of example 2D FT

### 3. Train Classifier

Run PCA and DFA to create discriminant functions:

```bash
python PCA_DFA.py
```

**Outputs**:
- `masking_parameters.pkl`: Discriminant functions and classification parameters
- `training_database.png`: Training database visualization
- `pca_scores.png`: PCA scores scatter plot
- `dfa_outcome.png`: DFA classification space
- `df_spectra.png`: Discriminant spectra
- `pca_decay.png`: PCA eigenvalue decay curve

### 4. Analyze New Recordings

Run CCP analysis on audio files:

```bash
# Process entire file (default)
python CCP.py data/29-06-23.wav

# Process specific time segment
python CCP.py data/29-06-23.wav \
    --start-min 15 --start-sec 6 \
    --end-min 15 --end-sec 15

# Process from start time to end of file
python CCP.py data/29-06-23.wav --start-min 10 --start-sec 0

# Custom output filename
python CCP.py data/my_recording.wav --output my_results.pkl
```

**Output**: `29-06-23.pkl` (DF scores for each time window)

**Note**: The time range you specify here determines what will be classified in the next step!

---

### Approach A: Geometric Polygon Classification

#### 5a. Analyze New Recordings

Update the audio file path in `CCP.py` or use it directly:

```bash
python CCP.py
```

**Output**: `29-06-23.pkl` (DF scores for each time window)

#### 6a. Classify Using Polygons

Option 1: Using `masking.py`:
```bash
python masking.py
```

Option 2: Using `test_recordings.py` with additional features:
```bash
# Test single file
python test_recordings.py --file 29-06-23.pkl

# Compare multiple files
python test_recordings.py --compare file1.pkl file2.pkl file3.pkl

# Adjust boundary shrink factor
python test_recordings.py --file 29-06-23.pkl --shrink 0.2
```

**Outputs**:
- `polygonal_areas.pkl`: Boundary polygon coordinates
- `polygonal_areas.png`: Training data visualization
- `classification_result.png`: Classification boundaries and test points
- `*_timeline.png`: Timeline visualization of classifications
- `*_test_results.pkl`: Detailed test results
- `comparison_results.png`: Multi-file comparison (if comparing)

---

### Approach B: Traditional ML Classification

#### 5b. Train ML Classifiers

Train and compare multiple ML classifiers:

```bash
python train_ml_classifiers.py
```

This will:
- Train LDA, SVM, Random Forest, Gradient Boosting, and Neural Network classifiers
- Perform 5-fold cross-validation
- Select the best model
- Save the best model to `trained_ml_classifier.pkl`

**Outputs**:
- `trained_ml_classifier.pkl`: Best trained model with scaler
- `classifier_comparison.png`: Performance comparison bar chart
- `confusion_matrix_best.png`: Confusion matrix for best classifier

#### 6b. Classify Using ML Model

Classify new audio directly without CCP step:

```bash
# Classify entire audio file (default behavior)
python classify_with_ml.py data/29-06-23.wav

# Classify a specific time segment
python classify_with_ml.py data/29-06-23.wav \
    --start-min 15 --start-sec 6 \
    --end-min 15 --end-sec 15 \
    --increment 1.0

# Classify from start to specific end time
python classify_with_ml.py data/29-06-23.wav \
    --end-min 10 --end-sec 0

# Custom output filename
python classify_with_ml.py data/my_recording.wav \
    --output my_results.pkl
```

**Note**: If `--end-min` and `--end-sec` are not specified (or both are 0), the script processes the entire file from the start time.

**Outputs**:
- `*_ml_classified.pkl`: Classification results with probabilities
- `ml_classification_result.png`: Timeline and probability visualizations

---

### Comparing Both Approaches

#### Step 1: Run Both Classifications on Same Segment

```bash
# Define the segment to analyze (example: 15:06 to 15:15)
START_MIN=15
START_SEC=6
END_MIN=15
END_SEC=15

# Approach A (Polygon)
# Step 1a: Compute DF scores with CCP
python CCP.py data/29-06-23.wav \
    --start-min $START_MIN --start-sec $START_SEC \
    --end-min $END_MIN --end-sec $END_SEC

# Step 1b: Classify using polygons (produces 29-06-23_test_results.pkl)
python test_recordings.py --file 29-06-23.pkl

# Approach B (ML)
# Step 2: Train ML model (if not already trained)
python train_ml_classifiers.py

# Step 3: Classify with ML using SAME segment (produces 29-06-23_ml_classified.pkl)
python classify_with_ml.py data/29-06-23.wav \
    --start-min $START_MIN --start-sec $START_SEC \
    --end-min $END_MIN --end-sec $END_SEC
```

**Alternative: Process Entire File**

```bash
# Both approaches on entire file
python CCP.py data/29-06-23.wav  # Entire file
python test_recordings.py --file 29-06-23.pkl
python classify_with_ml.py data/29-06-23.wav  # Entire file
```

**Critical**: Both approaches **must process the same time segment** for meaningful comparison!

#### Step 2: Compare Results Visually

```bash
# Create side-by-side comparison in DF space
python compare_classifications.py \
    --ccp-file 29-06-23.pkl \
    --polygon-file 29-06-23_test_results.pkl \
    --ml-file 29-06-23_ml_classified.pkl
```

This generates:
- `classification_comparison.png` - Side-by-side visualization showing:
  - **Left**: Polygon classification with boundaries and test points (circles)
  - **Right**: ML classification with reference boundaries and test points (squares)
- Console output with agreement statistics:
  - Overall agreement percentage
  - Class-by-class counts
  - Detailed disagreement analysis

**Example Output:**
```
Classification Agreement Analysis
============================================================
Total test points: 8
Polygon ambiguous classifications: 0
Valid classifications: 8
Agreement: 7/8 (87.5%)
Disagreement: 1/8 (12.5%)

Class-by-class counts:
Class           Polygon    ML
-----------------------------------
Hornet          2          2
Bee             4          5
Winter BG       1          0
Summer BG       1          1
Ambiguous       0          -
```

**Key Differences**:
- **Polygon Approach**: Interpretable, visualizable decision boundaries, requires CCP preprocessing, can have ambiguous cases
- **ML Approach**: Direct classification, probability estimates, potentially higher accuracy, no ambiguous category, more black-box
- **Agreement**: Typically 80-95% agreement on clear cases; differences appear at class boundaries

## Parameters

Key processing parameters (defined in scripts):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `window_length` | 0.5 s | Window length for training samples |
| `mf` | 4 Hz | Spectral repetition parameter |
| `tr` | 0.04 s | Time resolution |
| `increment_sec` | 1.0 s | Window increment for CCP analysis |
| `freq_range` | 4:60 | Frequency index range (~100-1500 Hz) |
| `n_components` | 9 | Number of PCA components for DFA |
| `shrink_factor` | 0.1 | Boundary polygon shrink factor |

## Classification Codes

The system assigns the following numerical codes:

- **999**: Hornet detection
- **998**: Bee detection
- **997**: Winter background noise
- **996**: Summer background noise
- **995**: No category (ambiguous/overlapping)

## Key Differences from MATLAB

1. **Data Storage**: Uses Python pickle (`.pkl`) instead of MATLAB `.mat` files
2. **Audio Loading**: Uses `librosa` instead of MATLAB's `audioread`
3. **Linear Algebra**: Uses `numpy` and `scipy` instead of MATLAB's built-in functions
4. **Machine Learning**: Uses `scikit-learn` for PCA and LDA instead of custom MATLAB functions
5. **Polygon Operations**: Uses `matplotlib.path` for point-in-polygon testing instead of MATLAB's `inpolygon`
6. **Visualization**: Uses `matplotlib` instead of MATLAB plotting

## Data Files

The pipeline creates the following intermediate files:

| File | Description | Used By |
|------|-------------|---------|
| `sound_files.pkl` | Preprocessed training audio recordings | Both approaches |
| `fourth_TDB.pkl` | Training database with 2D Fourier Transforms | Both approaches |
| `masking_parameters.pkl` | Discriminant functions and centroids | Approach A |
| `polygonal_areas.pkl` | Classification boundary polygons | Approach A |
| `*.pkl` (from CCP) | Test results (DF scores) | Approach A |
| `*_classified.pkl` | Polygon classification results | Approach A |
| `*_test_results.pkl` | Detailed test results with statistics | Approach A |
| `trained_ml_classifier.pkl` | Trained ML model with scaler | Approach B |
| `*_ml_classified.pkl` | ML classification results with probabilities | Approach B |

## Example Usage

### Complete Pipeline - Approach A (Polygon Classification)

```bash
# 1. Load and preprocess audio
python read_sound_files.py

# 2. Create training database
python TDB.py

# 3. Train PCA/DFA
python PCA_DFA.py

# 4. Analyze test recording with CCP
python CCP.py

# 5. Classify using polygons
python test_recordings.py --file 29-06-23.pkl
```

### Complete Pipeline - Approach B (ML Classification)

```bash
# 1. Load and preprocess audio
python read_sound_files.py

# 2. Create training database
python TDB.py

# 3. Train ML classifiers
python train_ml_classifiers.py

# 4. Classify test recording directly
python classify_with_ml.py data/29-06-23.wav \
    --start-min 15 --start-sec 6 \
    --end-min 15 --end-sec 15
```

## Approach Comparison

| Feature | Polygon Classification (A) | ML Classification (B) |
|---------|---------------------------|----------------------|
| **Training** | PCA + DFA (creates templates) | Multiple ML algorithms |
| **Classification** | Template matching + geometry | Direct prediction |
| **Preprocessing** | Requires CCP.py step | Built into classify script |
| **Outputs** | DF scores → polygon membership | Class labels + probabilities |
| **Interpretability** | High (visualizable boundaries) | Medium (depends on model) |
| **Speed** | Fast after CCP | Fast (no CCP needed) |
| **Ambiguous Cases** | Category 995 (no polygon) | Low probability predictions |
| **Best For** | Research, visualization, understanding | Production, accuracy, automation |

## Troubleshooting

### Missing audio files
- Ensure audio files are in `data/audio/` directory
- Update file paths in scripts to match your directory structure

### Import errors
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### Memory issues
- Large audio files may require significant RAM
- Process shorter segments or reduce the number of training samples

### Classification accuracy
- Adjust `shrink_factor` in classification scripts (0.0 = tight, 1.0 = loose)
- Verify training data timings are correct
- Ensure audio quality is sufficient

## License

This is a translation of the original MATLAB code. Please refer to the original repository for licensing information:
https://github.com/HThomasntu/hornet_TDB_MLA_matlab

## Citation

If you use this code, please cite the original research paper:
[Paper citation to be added]

## Acknowledgments

- Original MATLAB implementation by the authors of the research paper
- Python translation created for improved accessibility and integration with modern ML workflows
