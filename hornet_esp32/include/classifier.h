/**
 * Hornet Classifier
 * Polygon-based classification using discriminant function scores
 */

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <Arduino.h>

// Classification result structure
struct ClassificationResult {
    uint16_t class_code;      // CLASS_HORNET, CLASS_BEE, etc.
    const char* class_name;   // Human-readable name
    float df_score_1;         // Discriminant function score 1
    float df_score_2;         // Discriminant function score 2
    bool is_hornet;           // Quick check for hornet detection
};

class HornetClassifier {
public:
    HornetClassifier();

    /**
     * Compute discriminant function scores from 2D FT spectrum
     *
     * @param spectrum_2d Flattened 2D FT spectrum (56 x 49 = 2744 elements)
     * @param df1_out Output for DF score 1
     * @param df2_out Output for DF score 2
     */
    void computeDFScores(const float* spectrum_2d, float& df1_out, float& df2_out);

    /**
     * Classify a point based on DF scores using polygon boundaries
     *
     * @param df1 Discriminant function score 1
     * @param df2 Discriminant function score 2
     * @return Classification result
     */
    ClassificationResult classify(float df1, float df2);

private:
    /**
     * Test if point is inside polygon using ray casting algorithm
     *
     * @param x Point x coordinate
     * @param y Point y coordinate
     * @param poly_x Polygon x coordinates
     * @param poly_y Polygon y coordinates
     * @param n_vertices Number of polygon vertices
     * @return true if point is inside polygon
     */
    bool pointInPolygon(float x, float y, const float* poly_x, const float* poly_y, uint8_t n_vertices);
};

#endif // CLASSIFIER_H
