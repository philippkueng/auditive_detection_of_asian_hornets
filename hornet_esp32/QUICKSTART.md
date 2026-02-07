# ESP32 Hornet Detection - Quick Start Guide

## 5-Minute Setup

### 1. Hardware Setup (2 minutes)

**Wire INMP441 microphone to ESP32:**

```
INMP441  ->  ESP32
-------------------
VDD      ->  3.3V
GND      ->  GND
SD       ->  GPIO32
WS       ->  GPIO15
SCK      ->  GPIO14
L/R      ->  GND
```

### 2. Export Model (1 minute)

```bash
cd hornet_python
python export_model_to_cpp.py
```

### 3. Upload to ESP32 (2 minutes)

```bash
cd ../hornet_esp32
pio run --target upload
pio device monitor
```

## Expected Output

```
================================
Hornet Detection System for ESP32
================================

System initialized successfully!
Starting detection...

Detection #1
DF Scores: (0.45, -0.12)
Classification: Bee

Detection #2
DF Scores: (1.23, 0.57)
Classification: HORNET

!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!  HORNET DETECTED!!!  !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "I2S read failed" | Check wiring, especially SD/WS/SCK pins |
| Out of memory | Enable PSRAM in platformio.ini (already done) |
| Slow processing | Set CPU to 240MHz: `board_build.f_cpu = 240000000L` |
| Wrong classifications | Re-train model or adjust microphone placement |

## Next Steps

1. Test with known hornet/bee sounds
2. Adjust detection interval in `main.cpp`
3. Add LED indicator (see README)
4. Add WiFi/MQTT for remote alerts
5. Deploy in field for real-world testing

## Key Files

- `include/model_data.h` - Generated model (don't edit manually)
- `src/main.cpp` - Main program (edit detection settings here)
- `include/audio_processor.h` - Microphone pins (edit if needed)

## Support

Full documentation: See `README.md`

Hardware issues: Check wiring diagram in README

Model issues: Retrain in `hornet_python/` first
