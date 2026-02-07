/**
 * Audio Processor
 * Handles audio capture from I2S microphone and 2D FT computation
 */

#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <Arduino.h>
#include <driver/i2s.h>
#include <arduinoFFT.h>

// I2S configuration
#define I2S_WS 15    // Word Select (LRCLK)
#define I2S_SD 32    // Serial Data (DOUT)
#define I2S_SCK 14   // Serial Clock (BCLK)

// Audio buffer size
#define SAMPLE_RATE 48000
#define BUFFER_SIZE 1024
#define WINDOW_SIZE 48000  // 1 second window

class AudioProcessor {
public:
    AudioProcessor();
    ~AudioProcessor();

    /**
     * Initialize I2S microphone
     * @return true if successful
     */
    bool begin();

    /**
     * Capture 1 second of audio and compute 2D FT
     *
     * @param spectrum_out Output buffer for flattened 2D spectrum (56x49 = 2744 floats)
     *                     Must be pre-allocated
     * @return true if successful
     */
    bool captureAndProcess(float* spectrum_out);

private:
    int16_t* audio_buffer;
    float* window_buffer;
    bool initialized;

    /**
     * Read audio samples from I2S microphone
     *
     * @param buffer Output buffer for samples
     * @param num_samples Number of samples to read
     * @return true if successful
     */
    bool readAudio(int16_t* buffer, size_t num_samples);

    /**
     * Compute simplified 2D Fourier Transform
     * This computes STFT followed by FFT along time axis
     *
     * @param audio Audio samples
     * @param num_samples Number of audio samples
     * @param spectrum_out Output 2D spectrum (will be cropped to 56x49)
     * @return true if successful
     */
    bool compute2DFT(int16_t* audio, size_t num_samples, float* spectrum_out);
};

#endif // AUDIO_PROCESSOR_H
