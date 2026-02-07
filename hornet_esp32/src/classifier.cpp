/**
 * Hornet Classifier Implementation
 */

#include "classifier.h"
#include "model_data.h"

HornetClassifier::HornetClassifier() {
    // Constructor
}

void HornetClassifier::computeDFScores(const float* spectrum_2d, float& df1_out, float& df2_out) {
    // Compute DF scores as dot product of spectrum with discriminant spectra
    // spectrum_2d should be 56 x 49 = 2744 elements (cropped and scaled)

    df1_out = 0.0f;
    df2_out = 0.0f;

    uint16_t spectrum_size = DF_SPECTRUM_1_rows * DF_SPECTRUM_1_cols;

    for (uint16_t i = 0; i < spectrum_size; i++) {
        // Read from PROGMEM (flash)
        float df1_val = pgm_read_float(&DF_SPECTRUM_1[i]);
        float df2_val = pgm_read_float(&DF_SPECTRUM_2[i]);

        df1_out += spectrum_2d[i] * df1_val;
        df2_out += spectrum_2d[i] * df2_val;
    }
}

ClassificationResult HornetClassifier::classify(float df1, float df2) {
    ClassificationResult result;
    result.df_score_1 = df1;
    result.df_score_2 = df2;

    // Check each polygon in order of priority
    // Read polygon coordinates from PROGMEM

    // Check hornet polygon
    bool in_hornet = false;
    if (POLYGON_HORNET_SIZE >= 3) {
        float* hornet_x = new float[POLYGON_HORNET_SIZE];
        float* hornet_y = new float[POLYGON_HORNET_SIZE];
        for (uint8_t i = 0; i < POLYGON_HORNET_SIZE; i++) {
            hornet_x[i] = pgm_read_float(&POLYGON_HORNET_X[i]);
            hornet_y[i] = pgm_read_float(&POLYGON_HORNET_Y[i]);
        }
        in_hornet = pointInPolygon(df1, df2, hornet_x, hornet_y, POLYGON_HORNET_SIZE);
        delete[] hornet_x;
        delete[] hornet_y;
    }

    // Check bee polygon
    bool in_bee = false;
    if (POLYGON_BEE_SIZE >= 3) {
        float* bee_x = new float[POLYGON_BEE_SIZE];
        float* bee_y = new float[POLYGON_BEE_SIZE];
        for (uint8_t i = 0; i < POLYGON_BEE_SIZE; i++) {
            bee_x[i] = pgm_read_float(&POLYGON_BEE_X[i]);
            bee_y[i] = pgm_read_float(&POLYGON_BEE_Y[i]);
        }
        in_bee = pointInPolygon(df1, df2, bee_x, bee_y, POLYGON_BEE_SIZE);
        delete[] bee_x;
        delete[] bee_y;
    }

    // Check winter background polygon
    bool in_winter = false;
    if (POLYGON_WINTER_SIZE >= 3) {
        float* winter_x = new float[POLYGON_WINTER_SIZE];
        float* winter_y = new float[POLYGON_WINTER_SIZE];
        for (uint8_t i = 0; i < POLYGON_WINTER_SIZE; i++) {
            winter_x[i] = pgm_read_float(&POLYGON_WINTER_X[i]);
            winter_y[i] = pgm_read_float(&POLYGON_WINTER_Y[i]);
        }
        in_winter = pointInPolygon(df1, df2, winter_x, winter_y, POLYGON_WINTER_SIZE);
        delete[] winter_x;
        delete[] winter_y;
    }

    // Check summer background polygon
    bool in_summer = false;
    if (POLYGON_SUMMER_SIZE >= 3) {
        float* summer_x = new float[POLYGON_SUMMER_SIZE];
        float* summer_y = new float[POLYGON_SUMMER_SIZE];
        for (uint8_t i = 0; i < POLYGON_SUMMER_SIZE; i++) {
            summer_x[i] = pgm_read_float(&POLYGON_SUMMER_X[i]);
            summer_y[i] = pgm_read_float(&POLYGON_SUMMER_Y[i]);
        }
        in_summer = pointInPolygon(df1, df2, summer_x, summer_y, POLYGON_SUMMER_SIZE);
        delete[] summer_x;
        delete[] summer_y;
    }

    // Classify based on exclusive membership
    if (in_hornet && !in_bee && !in_winter && !in_summer) {
        result.class_code = CLASS_HORNET;
        result.class_name = "HORNET";
        result.is_hornet = true;
    } else if (!in_hornet && in_bee && !in_winter && !in_summer) {
        result.class_code = CLASS_BEE;
        result.class_name = "Bee";
        result.is_hornet = false;
    } else if (!in_hornet && !in_bee && in_winter && !in_summer) {
        result.class_code = CLASS_WINTER_BG;
        result.class_name = "Winter BG";
        result.is_hornet = false;
    } else if (!in_hornet && !in_bee && !in_winter && in_summer) {
        result.class_code = CLASS_SUMMER_BG;
        result.class_name = "Summer BG";
        result.is_hornet = false;
    } else {
        result.class_code = CLASS_AMBIGUOUS;
        result.class_name = "Ambiguous";
        result.is_hornet = false;
    }

    return result;
}

bool HornetClassifier::pointInPolygon(float x, float y, const float* poly_x, const float* poly_y, uint8_t n_vertices) {
    // Ray casting algorithm
    bool inside = false;

    for (uint8_t i = 0, j = n_vertices - 1; i < n_vertices; j = i++) {
        float xi = poly_x[i], yi = poly_y[i];
        float xj = poly_x[j], yj = poly_y[j];

        bool intersect = ((yi > y) != (yj > y)) &&
                         (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

        if (intersect) {
            inside = !inside;
        }
    }

    return inside;
}
