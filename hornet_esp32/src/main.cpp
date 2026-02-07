/**
 * Hornet Detection System for ESP32
 * Main program
 */

#include <Arduino.h>
#include "audio_processor.h"
#include "classifier.h"
#include "model_data.h"

// Global objects
AudioProcessor audioProcessor;
HornetClassifier classifier;

// Processing buffer
float* spectrum_buffer = nullptr;

// Detection settings
#define DETECTION_INTERVAL_MS 2000  // Check every 2 seconds
#define HORNET_ALERT_COOLDOWN_MS 5000  // Don't spam alerts

unsigned long last_detection_time = 0;
unsigned long last_alert_time = 0;
uint32_t total_detections = 0;
uint32_t hornet_detections = 0;

void setup() {
    // Initialize serial
    Serial.begin(115200);
    while (!Serial && millis() < 3000);  // Wait up to 3 seconds for serial

    Serial.println("\n\n");
    Serial.println("================================");
    Serial.println("Hornet Detection System for ESP32");
    Serial.println("================================");
    Serial.println();

    // Print model information
    Serial.println("Model Configuration:");
    Serial.printf("  Sample Rate: %d Hz\n", MODEL_SAMPLE_RATE);
    Serial.printf("  Window Length: %.2f s\n", MODEL_WINDOW_LENGTH);
    Serial.printf("  Spectral Repetition (mf): %.2f Hz\n", MODEL_MF);
    Serial.printf("  Time Resolution (tr): %.4f s\n", MODEL_TR);
    Serial.printf("  Frequency Range: %d - %d\n", MODEL_FREQ_MIN, MODEL_FREQ_MAX);
    Serial.printf("  Spectrum Size: %d x %d\n", DF_SPECTRUM_1_rows, DF_SPECTRUM_1_cols);
    Serial.println();

    // Allocate spectrum buffer
    spectrum_buffer = new float[DF_SPECTRUM_1_rows * DF_SPECTRUM_1_cols];
    if (!spectrum_buffer) {
        Serial.println("ERROR: Failed to allocate spectrum buffer");
        Serial.println("System halted");
        while (1) delay(1000);
    }

    // Initialize audio processor
    if (!audioProcessor.begin()) {
        Serial.println("ERROR: Failed to initialize audio processor");
        Serial.println("Check microphone wiring:");
        Serial.printf("  WS (LRCLK)  -> GPIO %d\n", I2S_WS);
        Serial.printf("  SD (DOUT)   -> GPIO %d\n", I2S_SD);
        Serial.printf("  SCK (BCLK)  -> GPIO %d\n", I2S_SCK);
        Serial.printf("  VDD         -> 3.3V\n");
        Serial.printf("  GND         -> GND\n");
        Serial.println("System halted");
        while (1) delay(1000);
    }

    Serial.println("System initialized successfully!");
    Serial.println("Starting detection...");
    Serial.println();

    delay(1000);  // Let things stabilize
}

void loop() {
    unsigned long current_time = millis();

    // Check if it's time for another detection
    if (current_time - last_detection_time >= DETECTION_INTERVAL_MS) {
        last_detection_time = current_time;

        Serial.println("----------------------------------------");
        Serial.printf("Detection #%d (Time: %lu ms)\n", total_detections + 1, current_time);
        Serial.println("----------------------------------------");

        // Capture audio and compute 2D FT
        if (!audioProcessor.captureAndProcess(spectrum_buffer)) {
            Serial.println("ERROR: Audio processing failed");
            Serial.println();
            return;
        }

        // Compute discriminant function scores
        float df1, df2;
        classifier.computeDFScores(spectrum_buffer, df1, df2);

        Serial.printf("DF Scores: (%.4f, %.4f)\n", df1, df2);

        // Classify
        ClassificationResult result = classifier.classify(df1, df2);

        Serial.printf("Classification: %s (code: %d)\n", result.class_name, result.class_code);

        // Update statistics
        total_detections++;

        // Check for hornet detection
        if (result.is_hornet) {
            hornet_detections++;

            // Alert if not in cooldown
            if (current_time - last_alert_time >= HORNET_ALERT_COOLDOWN_MS) {
                last_alert_time = current_time;

                Serial.println();
                Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                Serial.println("!!!   HORNET DETECTED - ALERT!!!    !!!");
                Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                Serial.printf("!!!   DF Scores: (%.4f, %.4f)     !!!\n", df1, df2);
                Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                Serial.println();
            } else {
                Serial.println("  >> Hornet detected (alert in cooldown)");
            }
        }

        // Print statistics
        Serial.printf("Statistics: %d total, %d hornets (%.1f%%)\n",
                      total_detections,
                      hornet_detections,
                      100.0f * hornet_detections / total_detections);

        Serial.println();

        // Free heap status
        Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
        Serial.println();
    }

    // Small delay to prevent watchdog issues
    delay(10);
}
