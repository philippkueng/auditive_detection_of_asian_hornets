## Hornet Detection System for ESP32

Real-time hornet detection using polygon-based classification on ESP32 microcontroller with I2S MEMS microphone.

## Overview

This system runs the trained polygon-based classification model directly on an ESP32, capturing audio from a microphone and detecting Asian hornets in real-time. When a hornet is detected, an alert is logged to the serial console.

## Hardware Requirements

### Required Components

1. **ESP32 Development Board**
   - ESP32-DevKitC or compatible
   - Minimum 4MB flash
   - PSRAM recommended for better performance

2. **I2S MEMS Microphone** (Recommended: INMP441)
   - Better audio quality than analog microphones
   - Direct digital interface
   - Cost: ~$2-5 USD
   - Alternative: MAX4466, MAX9814 (analog, requires ADC)

3. **Cables & Breadboard**
   - Jumper wires
   - Breadboard (optional)
   - USB cable for programming and power

### Microphone Wiring (INMP441)

```
INMP441 Pin    ->    ESP32 Pin    ->    Description
---------------------------------------------------------
VDD            ->    3.3V          ->    Power supply
GND            ->    GND           ->    Ground
SD (DOUT)      ->    GPIO 32       ->    Serial Data
WS (LRCLK)     ->    GPIO 15       ->    Word Select / Left-Right Clock
SCK (BCLK)     ->    GPIO 14       ->    Serial Clock / Bit Clock
L/R            ->    GND           ->    Left channel (can also connect to 3.3V for right)
```

### Wiring Diagram

```
         INMP441                    ESP32
     ┌─────────────┐           ┌──────────┐
     │             │           │          │
     │  VDD    ●───────────────●  3.3V    │
     │  GND    ●───────────────●  GND     │
     │  SD     ●───────────────●  GPIO32  │
     │  WS     ●───────────────●  GPIO15  │
     │  SCK    ●───────────────●  GPIO14  │
     │  L/R    ●───────────────●  GND     │
     │             │           │          │
     └─────────────┘           └──────────┘
```

**Important Notes:**
- Ensure stable 3.3V power supply (microphone is sensitive to voltage)
- Keep wires short to reduce noise
- Add a 100nF capacitor between VDD and GND near the microphone (optional but recommended)

## Software Setup

### Prerequisites

1. **PlatformIO** installed (VSCode extension or CLI)
   ```bash
   # Install PlatformIO CLI (if not using VSCode)
   pip install platformio
   ```

2. **Python 3.8+** (for model export)
   ```bash
   pip install numpy scipy
   ```

### Installation Steps

#### Step 1: Export Trained Model

First, export your trained Python model to C++ headers:

```bash
cd hornet_python

# Make sure you've trained the model first
python PCA_DFA.py
python masking.py  # Or test_recordings.py

# Export model to C++ header
python export_model_to_cpp.py
```

This creates `hornet_esp32/include/model_data.h` with:
- Discriminant function spectra
- Polygon boundary coordinates
- Processing parameters

#### Step 2: Build and Upload

```bash
cd ../hornet_esp32

# Build the project
pio run

# Upload to ESP32 (connect via USB)
pio run --target upload

# Open serial monitor
pio device monitor
```

**Alternative (VSCode):**
1. Open `hornet_esp32` folder in VSCode
2. PlatformIO: Build (Ctrl+Alt+B)
3. PlatformIO: Upload (Ctrl+Alt+U)
4. PlatformIO: Serial Monitor (Ctrl+Alt+S)

## Usage

### Serial Monitor Output

Once uploaded and running, you'll see output like:

```
================================
Hornet Detection System for ESP32
================================

Model Configuration:
  Sample Rate: 48000 Hz
  Window Length: 1.00 s
  Spectral Repetition (mf): 4.00 Hz
  Time Resolution (tr): 0.0400 s
  Frequency Range: 4 - 60
  Spectrum Size: 56 x 49

System initialized successfully!
Starting detection...

----------------------------------------
Detection #1 (Time: 2134 ms)
----------------------------------------
Capturing audio...
Audio captured, processing...
Computing 2D FT...
  FFT size: 1024, Frames: 48
  Frame 0/48
  Frame 10/48
  ...
  STFT complete, computing modulation spectrum...
  Modulation spectrum complete
  Spectrum normalized (max: 1245.67)
Processing complete
DF Scores: (0.4521, -0.1234)
Classification: Bee (code: 998)
Statistics: 1 total, 0 hornets (0.0%)

Free Heap: 234512 bytes

----------------------------------------
Detection #2 (Time: 4267 ms)
----------------------------------------
...
DF Scores: (1.2341, 0.5678)
Classification: HORNET (code: 999)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!   HORNET DETECTED - ALERT!!!    !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!   DF Scores: (1.2341, 0.5678)   !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Statistics: 2 total, 1 hornets (50.0%)
```

### Detection Behavior

- **Detection Interval**: Checks every 2 seconds (configurable in `main.cpp`)
- **Alert Cooldown**: 5 seconds between hornet alerts (prevents spam)
- **Classification**: Hornet, Bee, Winter Background, Summer Background, or Ambiguous
- **Statistics**: Tracks total detections and hornet detection rate

## Configuration

### Adjusting Detection Settings

Edit `src/main.cpp`:

```cpp
// Detection interval (milliseconds)
#define DETECTION_INTERVAL_MS 2000  // Check every 2 seconds

// Alert cooldown (milliseconds)
#define HORNET_ALERT_COOLDOWN_MS 5000  // 5 seconds between alerts
```

### Changing Microphone Pins

Edit `include/audio_processor.h`:

```cpp
#define I2S_WS 15    // Word Select (LRCLK)
#define I2S_SD 32    // Serial Data (DOUT)
#define I2S_SCK 14   // Serial Clock (BCLK)
```

Rebuild and upload after making changes.

## Performance

### Memory Usage

- **Program Storage**: ~300-500 KB (depends on model size)
- **RAM Usage**: ~150-200 KB peak during processing
- **Model Data**: ~15-20 KB (stored in flash/PROGMEM)

### Processing Time

- **Audio Capture**: ~1 second (capturing 1-second window)
- **2D FT Computation**: ~3-5 seconds (depends on ESP32 speed)
- **Classification**: <10 ms
- **Total per Detection**: ~4-6 seconds

### Optimization Tips

1. **Enable PSRAM**: Provides more RAM for buffers
2. **Overclock CPU**: Set CPU frequency to 240 MHz in platformio.ini:
   ```ini
   board_build.f_cpu = 240000000L
   ```
3. **Reduce FFT Size**: Modify `fft_size` in `audio_processor.cpp` (affects accuracy)
4. **Increase Detection Interval**: Reduce CPU load by checking less frequently

## Troubleshooting

### No Audio Captured

**Symptoms**: "ERROR: I2S read failed" or zeros in audio buffer

**Solutions**:
1. Check microphone wiring (especially SD, WS, SCK)
2. Verify 3.3V power supply is stable
3. Ensure microphone is not damaged
4. Try different GPIO pins
5. Add pull-up resistors (4.7kΩ) on I2S data lines

### Out of Memory Errors

**Symptoms**: "ERROR: Failed to allocate..." or random crashes

**Solutions**:
1. Use ESP32 with PSRAM
2. Enable PSRAM in platformio.ini (already enabled)
3. Reduce FFT size in `audio_processor.cpp`
4. Use external memory for model data

### Inaccurate Classifications

**Symptoms**: Too many false positives/negatives

**Solutions**:
1. Re-train model with more representative data
2. Ensure microphone placement is similar to training setup
3. Check audio quality (noise, clipping)
4. Adjust polygon boundaries in training
5. Verify sample rate matches training (48kHz)

### Slow Processing

**Symptoms**: Takes >10 seconds per detection

**Solutions**:
1. Set CPU frequency to 240 MHz
2. Reduce FFT size (trade-off: accuracy)
3. Simplify 2D FT computation
4. Use hardware acceleration (ESP-DSP library)
5. Optimize polygon check (reduce vertices)

### Serial Monitor Shows Garbage

**Symptoms**: Unreadable characters in serial output

**Solutions**:
1. Set baud rate to 115200 in serial monitor
2. Check USB cable and connection
3. Press ESP32 reset button
4. Re-upload firmware

## Advanced Features

### Adding LED Indicator

Add to `main.cpp`:

```cpp
#define LED_PIN 2  // Built-in LED

void setup() {
    pinMode(LED_PIN, OUTPUT);
    // ... rest of setup
}

// In loop(), when hornet detected:
if (result.is_hornet) {
    digitalWrite(LED_PIN, HIGH);  // Turn on LED
    delay(1000);
    digitalWrite(LED_PIN, LOW);   // Turn off LED
}
```

### WiFi/MQTT Alerts

Integrate WiFi and MQTT for remote alerting:

```cpp
#include <WiFi.h>
#include <PubSubClient.h>

// Add to main.cpp to publish detections to MQTT broker
```

### SD Card Logging

Log detections to SD card for later analysis:

```cpp
#include <SD.h>
#include <SPI.h>

// Log timestamp, DF scores, and classification to SD card
```

### Multiple Microphones

Use multiple I2S microphones on different buses (I2S_NUM_0, I2S_NUM_1) for spatial coverage.

## Project Structure

```
hornet_esp32/
├── platformio.ini          # PlatformIO configuration
├── include/
│   ├── model_data.h        # Generated model data (from Python)
│   ├── classifier.h        # Classification header
│   └── audio_processor.h   # Audio processing header
├── src/
│   ├── main.cpp            # Main program
│   ├── classifier.cpp      # Classification implementation
│   └── audio_processor.cpp # Audio processing implementation
└── README.md               # This file
```

## License

Same as the parent project. See LICENSE file in repository root.

## Acknowledgments

- Based on the Python translation of MATLAB hornet detection research
- Uses arduinoFFT library for FFT computation
- I2S driver from ESP-IDF

## Support

For issues specific to the ESP32 implementation:
1. Check this README's troubleshooting section
2. Verify hardware connections
3. Test with known audio samples
4. Open an issue on GitHub with serial monitor output

For issues with the underlying model or training:
- See the main hornet_python/README.md
- Re-train the model if needed
- Verify classification accuracy on PC first
