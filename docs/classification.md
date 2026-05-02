# Classification method

This document describes the classification pipeline used in this workspace and how reference images are processed into features.

## Overview

The classifier compares shape-based descriptors extracted from a cropped card image against a database of reference descriptors. The high-level flow is:

1. Crop or load a single-card image.
2. Detect the card color (y, r, g, b, or k for black).
3. Create a color-based mask and extract symbol contours.
4. Build a descriptor from the largest symbol contours.
5. Compare the descriptor to reference features and predict the label.

## Processing a card image

The main entry points are `classify_card()` and `classify_card_with_details()` in classify.py. They call `process_card_image()` and then compute a descriptor from the resulting contours.

### 1. Card color detection

- The cropped RGB image is downscaled (0.35x) for faster color classification.
- A color mask is computed for each candidate color using HSV thresholds (or RGB for black):
  - y, r, g, b use HSV ranges.
  - k uses an RGB threshold to detect dark cards.
- The first mask that covers more than 10% of the image area determines the card color.

### 2. Mask creation and contour extraction

- The color-specific threshold mask is filled by morphological closing.
- An optional morphological opening step can be applied to clean small artifacts.
- The mask is applied to the original RGB image to isolate the card symbol.
- Up to 11 contours are extracted from the mask. (for an ideal 8/skip card: 2 corner + 3 symbol outlines + 3x2 symbol holes)
- For debugging, threshold, mask, and contour images can be saved.

## Descriptor construction

Descriptors are computed in `compute_descriptor_from_contours()` from up to `max_symbol_contours` largest contours (by area).

For each selected contour:

- **Fourier descriptor**
  - The contour is resampled to `num_points` points.
  - A complex FFT is computed and truncated to `num_descriptors` coefficients.
  - The descriptor is made translation-, scale-, and rotation-invariant.
  - Magnitudes of the FFT coefficients are used as features.

- **Shape statistics**
  - Aspect ratio, extent, solidity, circularity, and Hu moments are extracted.

The per-contour features are aggregated:

- The Fourier features are averaged across contours.
- Shape features are summarized by mean and standard deviation.
- Structural features across contours are appended: contour count, area ratios, area variability, perimeter normalization, centroid spread, and coverage.

The final descriptor dimension is:

- `num_descriptors` (Fourier) + `2 * SHAPE_FEATURE_DIM` (mean + std) + `STRUCT_FEATURE_DIM`.

## Reference feature generation

Reference images are processed in two stages:

### 1. Cropping reference cards

`prepare_reference_images.py` crops known card locations from raw photos defined in `REFERENCE_IMAGES` and saves them to:

- data/reference_images/cropped

Each crop is named by its label, for example `y_3.jpg`, `r_reverse.jpg`, or `wild.jpg`.

### 2. Computing reference descriptors

`process_reference_images.py` calls `compute_reference_features()` to build the feature database:

- Each cropped reference card is processed by the same pipeline used for test cards.
- A descriptor is computed and stored alongside the label.
- Optional data augmentation is applied by splitting each reference image into halves (left, right, top, bottom), which adds label variants like `y_3_a` or `y_3_top`.

Finally, the following data are saved to:

- data/reference_images/reference_features.npz

This file contains labels, features, descriptor settings, and per-feature mean/std for scaling.

## Matching and prediction

`classify_descriptor_with_details()` performs the matching:

1. **Feature scaling**
   - The descriptor and reference features are standardized using the stored mean/std.

2. **Candidate filtering by color**
   - If the card is black (k), only `wild` and `draw_4` references are considered.
   - Otherwise, black-only references are excluded.

3. **Distance computation**
   - L2 (Euclidean) distances are computed between the query descriptor and each candidate feature vector.
   - Each feature vector is first standardized, so every dimension is in comparable units (z-scores).
   - The distance is a single scalar that measures overall dissimilarity in descriptor space: smaller means the shapes and structure are more similar.
   - If multiple variants exist, the best (smallest) distance per base label is kept.

4. **Prediction and confidence**
   - The top-1 base label is used as the primary prediction.
   - A confidence score is derived from the ratio of the best and second-best distances:
     - $confidence = 1 - best/second$

5. **Voting fallback**
   - If confidence is below `vote_min_conf`, the top-k labels vote on the value.
   - If a value appears at least `vote_min_count` times, it overrides the top-1 decision.

The final output label is in the format:

- `color_value` (example: `r_5`, `g_skip`) or
- `wild` / `draw_4` for black cards.

## Configuration

Matching parameters are configurable via config.json under the `matching` key:

- `top_k` (default 5)
- `vote_min_conf` (default 0.15)
- `vote_min_count` (default 2)

## Files to check for implementation details

- classify.py: CLI and entry point for classification
- utils/process_utils.py: preprocessing, feature extraction, matching
- prepare_reference_images.py: reference cropping
- process_reference_images.py: reference feature generation
