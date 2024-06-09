# OD Segmentation

This repository contains scripts for OD (Optic Disc) Segmentation.

## Input Data

The input data is expected to be structured as follows:

- **Images folder**: Contains the following subfolders:
  - **OD_testing**: Results of manual OD image testing.
  - **OD_training**: Results of manual OD image training.
  - **testing**: Testing data.
  - **training**: Training data.

## Output Data

The output data is organized as follows:

- **Result_OD folder**: Contains the result of OD Segmentation.
- **Folder A**: Contains bounding box information.
- **Folder B**: Contains F-Score visualization:
  - Green represents true positives.
  - Yellow represents false positives.
  - Red represents false negatives.
