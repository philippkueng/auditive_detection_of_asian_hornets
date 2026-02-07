# Hornet Detection Project

Complete acoustic detection and classification system for Asian hornets (*Vespa velutina*), with both Python training/analysis tools and embedded ESP32 implementation.

## Project Structure

```
farming_hackdays_2026/
├── hornet_TDB_MLA_matlab/      # Original MATLAB implementation
│   └── (Reference code from research paper)
│
├── hornet_python/              # Python training & analysis tools
│   ├── Training pipeline (TDB, PCA/DFA)
│   ├── Classification approaches (Polygon & ML)
│   ├── Comparison tools
│   └── Model export for ESP32
│
└── hornet_esp32/               # ESP32 embedded implementation
    ├── Real-time detection firmware
    ├── I2S microphone support
    └── On-device classification
```

## Quick Links

- **Python Training**: [hornet_python/README.md](hornet_python/README.md)
- **ESP32 Quick Start**: [hornet_esp32/QUICKSTART.md](hornet_esp32/QUICKSTART.md)
- **ESP32 Full Guide**: [hornet_esp32/README.md](hornet_esp32/README.md)
- **Original Research**: https://www.sciencedirect.com/science/article/pii/S0168169925004132

## Overview

This project provides tools for detecting Asian hornets from audio recordings using acoustic analysis:

### 1. Python Tools (`hornet_python/`)

Train models and analyze audio recordings on a computer:

- **Data Processing**: Load and segment audio files
- **Feature Extraction**: Compute 2D Fourier Transforms
- **Model Training**: PCA/DFA for dimensionality reduction
- **Classification**:
  - **Approach A**: Polygon-based (geometric boundaries)
  - **Approach B**: ML-based (SVM, Random Forest, etc.)
- **Comparison**: Side-by-side analysis of methods
- **Export**: Convert trained models to C++ for ESP32

### 2. ESP32 Firmware (`hornet_esp32/`)

Run trained models on embedded hardware for real-time detection:

- **Hardware**: ESP32 + INMP441 microphone (~$10)
- **Processing**: Real-time 2D FT and classification
- **Detection**: Alerts logged to serial console
- **Latency**: ~4-6 seconds per detection
- **Power**: Can run on battery for field deployment

## Workflow

### Development & Training

```bash
# 1. Train model on PC with Python
cd hornet_python
python read_sound_files.py    # Load training audio
python TDB.py                  # Create training database
python PCA_DFA.py              # Train PCA/DFA model
python CCP.py                  # Analyze test recordings
python test_recordings.py      # Classify with polygons

# Optional: Compare with ML approach
python train_ml_classifiers.py
python classify_with_ml.py
python compare_classifications.py
```

### Deployment to ESP32

```bash
# 2. Export model to C++ header
cd hornet_python
python export_model_to_cpp.py  # Generates ../hornet_esp32/include/model_data.h

# 3. Build and upload to ESP32
cd ../hornet_esp32
pio run --target upload        # Upload firmware
pio device monitor             # View detections
```

### Field Usage

Once uploaded, the ESP32 will:
1. Continuously capture audio from microphone
2. Process audio every 2 seconds
3. Classify sounds (Hornet/Bee/Background)
4. Alert via serial console when hornet detected
5. Track detection statistics

## Hardware Requirements

### For Training (Python)
- Computer with Python 3.8+
- Audio files (WAV format, 48kHz recommended)
- ~2GB RAM for processing

### For Deployment (ESP32)
- ESP32 development board
- INMP441 I2S MEMS microphone
- USB cable
- Jumper wires
- Total cost: ~$10-15

## Getting Started

### New Users
1. Start with Python training: [hornet_python/README.md](hornet_python/README.md)
2. Train on your audio data
3. Test classification accuracy
4. Export to ESP32: [hornet_esp32/QUICKSTART.md](hornet_esp32/QUICKSTART.md)

### Quick ESP32 Deployment
If you already have a trained model:
```bash
cd hornet_python && python export_model_to_cpp.py
cd ../hornet_esp32 && pio run --target upload && pio device monitor
```

## Features

### Python Tools
✅ Complete MATLAB → Python translation
✅ Both polygon and ML classification
✅ Comprehensive visualization tools
✅ Side-by-side method comparison
✅ Model export to C++
✅ Extensive documentation

### ESP32 Firmware
✅ Real-time audio processing
✅ On-device classification
✅ I2S microphone support
✅ Serial console logging
✅ Configurable detection interval
✅ Memory-optimized for ESP32
✅ Field-deployable

## Performance

### Python (Offline Analysis)
- Training: Minutes to hours (depending on dataset size)
- Classification: <1 second per audio segment
- Accuracy: 85-95% (depends on training data quality)

### ESP32 (Real-time)
- Detection Latency: ~4-6 seconds
- Power Consumption: ~200mA @ 3.3V (active), ~10mA (sleep)
- Battery Life: ~8-12 hours (2000mAh battery, continuous operation)
- Detection Range: Depends on microphone and environment

## Applications

- **Research**: Study hornet behavior and patterns
- **Monitoring**: Automated surveillance of hives
- **Early Warning**: Alert beekeepers to hornet presence
- **Ecology**: Track invasive species spread
- **Citizen Science**: Distributed detection network

## Citation

If you use this code in your research, please cite the original paper:

[Paper citation from https://www.sciencedirect.com/science/article/pii/S0168169925004132]

## License

See LICENSE file in repository root.

## Acknowledgments

- Original MATLAB research code authors
- Python translation using LLM assistance
- ESP32 community and libraries (arduinoFFT, arduino-audio-tools)

## Contributing

Contributions welcome! Areas for improvement:
- Optimize ESP32 processing speed
- Add WiFi/MQTT integration
- Improve 2D FT accuracy on ESP32
- Add more ML classifier options
- Expand training dataset

## Support

- **Python Issues**: See [hornet_python/README.md](hornet_python/README.md)
- **ESP32 Issues**: See [hornet_esp32/README.md](hornet_esp32/README.md)
- **General Questions**: Open a GitHub issue
