/**
 * Audio Processor Implementation
 */

#include "audio_processor.h"
#include "model_data.h"
#include <math.h>

AudioProcessor::AudioProcessor() : initialized(false) {
    audio_buffer = nullptr;
    window_buffer = nullptr;
}

AudioProcessor::~AudioProcessor() {
    if (audio_buffer) delete[] audio_buffer;
    if (window_buffer) delete[] window_buffer;
}

bool AudioProcessor::begin() {
    Serial.println("Initializing I2S microphone...");

    // Allocate buffers
    audio_buffer = new int16_t[WINDOW_SIZE];
    window_buffer = new float[BUFFER_SIZE];

    if (!audio_buffer || !window_buffer) {
        Serial.println("ERROR: Failed to allocate audio buffers");
        return false;
    }

    // Configure I2S
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = BUFFER_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    // Install I2S driver
    esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("ERROR: I2S driver install failed: %d\n", err);
        return false;
    }

    // Configure I2S pins
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    err = i2s_set_pin(I2S_NUM_0, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("ERROR: I2S pin config failed: %d\n", err);
        return false;
    }

    // Start I2S
    i2s_start(I2S_NUM_0);

    initialized = true;
    Serial.println("I2S microphone initialized successfully");
    return true;
}

bool AudioProcessor::readAudio(int16_t* buffer, size_t num_samples) {
    size_t bytes_read = 0;
    size_t bytes_to_read = num_samples * sizeof(int32_t);  // I2S reads 32-bit

    int32_t* temp_buffer = new int32_t[num_samples];
    if (!temp_buffer) {
        Serial.println("ERROR: Failed to allocate temp buffer");
        return false;
    }

    // Read from I2S
    esp_err_t result = i2s_read(I2S_NUM_0, temp_buffer, bytes_to_read, &bytes_read, portMAX_DELAY);

    if (result != ESP_OK || bytes_read != bytes_to_read) {
        Serial.printf("ERROR: I2S read failed: %d, bytes: %d/%d\n", result, bytes_read, bytes_to_read);
        delete[] temp_buffer;
        return false;
    }

    // Convert 32-bit to 16-bit (take upper 16 bits)
    for (size_t i = 0; i < num_samples; i++) {
        buffer[i] = (int16_t)(temp_buffer[i] >> 16);
    }

    delete[] temp_buffer;
    return true;
}

bool AudioProcessor::compute2DFT(int16_t* audio, size_t num_samples, float* spectrum_out) {
    // Simplified 2D FT computation using FFT-based approach
    // This is a computationally lighter version for ESP32

    Serial.println("Computing 2D FT...");

    // Parameters
    const uint16_t fft_size = 1024;
    const uint16_t hop_size = 768;  // 75% overlap
    const uint16_t num_frames = (num_samples - fft_size) / hop_size + 1;

    Serial.printf("  FFT size: %d, Frames: %d\n", fft_size, num_frames);

    // Allocate FFT buffers
    double* vReal = new double[fft_size];
    double* vImag = new double[fft_size];

    if (!vReal || !vImag) {
        Serial.println("ERROR: Failed to allocate FFT buffers");
        return false;
    }

    // Initialize arduinoFFT
    ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, fft_size, SAMPLE_RATE);

    // Compute STFT (Short-Time Fourier Transform)
    // Store magnitude spectrogram
    float** spectrogram = new float*[num_frames];
    for (uint16_t i = 0; i < num_frames; i++) {
        spectrogram[i] = new float[fft_size / 2];
    }

    // Hann window
    for (uint16_t frame = 0; frame < num_frames; frame++) {
        size_t start_idx = frame * hop_size;

        // Apply window and copy to FFT buffer
        for (uint16_t i = 0; i < fft_size; i++) {
            if (start_idx + i < num_samples) {
                // Hann window
                double window = 0.5 * (1.0 - cos(2.0 * PI * i / (fft_size - 1)));
                vReal[i] = audio[start_idx + i] * window;
                vImag[i] = 0.0;
            } else {
                vReal[i] = 0.0;
                vImag[i] = 0.0;
            }
        }

        // Compute FFT
        FFT.compute(FFTDirection::Forward);
        FFT.complexToMagnitude();

        // Store magnitude spectrum
        for (uint16_t i = 0; i < fft_size / 2; i++) {
            spectrogram[frame][i] = (float)vReal[i];
        }

        // Progress indicator
        if (frame % 10 == 0) {
            Serial.printf("  Frame %d/%d\n", frame, num_frames);
        }
    }

    Serial.println("  STFT complete, computing modulation spectrum...");

    // Simplified modulation spectrum:
    // For each frequency bin, compute FFT along time axis
    // This gives us the spectral repetition (modulation) for each frequency

    const uint16_t freq_bins_out = 56;  // Cropped frequency range (4:60)
    const uint16_t mod_bins_out = 49;   // Modulation frequency bins

    // Zero output spectrum
    for (uint16_t i = 0; i < freq_bins_out * mod_bins_out; i++) {
        spectrum_out[i] = 0.0f;
    }

    // For computational efficiency, we'll do a simplified version:
    // Sample key frequency bins and compute modulation spectrum
    const uint16_t freq_step = (fft_size / 2) / freq_bins_out;

    for (uint16_t freq_idx = 0; freq_idx < freq_bins_out; freq_idx++) {
        uint16_t bin = (freq_idx + 4) * freq_step;  // Offset by 4 (MODEL_FREQ_MIN)

        if (bin >= fft_size / 2) break;

        // Extract time series for this frequency bin
        for (uint16_t t = 0; t < min((int)num_frames, (int)fft_size); t++) {
            vReal[t] = spectrogram[t][bin];
            vImag[t] = 0.0;
        }

        // Pad with zeros if needed
        for (uint16_t t = num_frames; t < fft_size; t++) {
            vReal[t] = 0.0;
            vImag[t] = 0.0;
        }

        // FFT along time axis
        FFT.compute(FFTDirection::Forward);
        FFT.complexToMagnitude();

        // Store first mod_bins_out values
        for (uint16_t mod_idx = 0; mod_idx < mod_bins_out && mod_idx < fft_size / 2; mod_idx++) {
            spectrum_out[freq_idx * mod_bins_out + mod_idx] = (float)vReal[mod_idx];
        }
    }

    Serial.println("  Modulation spectrum complete");

    // Normalize spectrum by maximum value
    float max_val = 0.0f;
    for (uint16_t i = 0; i < freq_bins_out * mod_bins_out; i++) {
        if (spectrum_out[i] > max_val) {
            max_val = spectrum_out[i];
        }
    }

    if (max_val > 0.0f) {
        for (uint16_t i = 0; i < freq_bins_out * mod_bins_out; i++) {
            spectrum_out[i] /= max_val;
        }
    }

    Serial.printf("  Spectrum normalized (max: %.2f)\n", max_val);

    // Cleanup
    delete[] vReal;
    delete[] vImag;
    for (uint16_t i = 0; i < num_frames; i++) {
        delete[] spectrogram[i];
    }
    delete[] spectrogram;

    return true;
}

bool AudioProcessor::captureAndProcess(float* spectrum_out) {
    if (!initialized) {
        Serial.println("ERROR: Audio processor not initialized");
        return false;
    }

    Serial.println("\nCapturing audio...");

    // Read 1 second of audio
    if (!readAudio(audio_buffer, WINDOW_SIZE)) {
        return false;
    }

    Serial.println("Audio captured, processing...");

    // Compute 2D FT
    if (!compute2DFT(audio_buffer, WINDOW_SIZE, spectrum_out)) {
        return false;
    }

    Serial.println("Processing complete");
    return true;
}
