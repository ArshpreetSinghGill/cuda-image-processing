# Configurable GPU Image Processing Pipeline using CUDA

## Overview

This project implements a configurable GPU-accelerated image processing pipeline using CUDA. It processes over 100 images from the USC SIPI dataset and applies multiple transformations using custom CUDA kernels.

The system supports multiple processing modes through command-line arguments, allowing dynamic selection of GPU-based operations such as edge detection, histogram equalization, and sharpening.

---

## Features

* Batch processing of 100+ images
* GPU acceleration using custom CUDA kernels
* Multiple processing modes:

  * Edge Detection (Grayscale + Blur + Sobel)
  * Histogram Equalization
  * Image Sharpening
* Command-line interface for flexible execution
* Performance timing for GPU operations

---

## Dataset

* USC SIPI Image Dataset
* ~100 images:

  * Textures (~64 images)
  * Miscellaneous (~39 images)

> Note: Dataset is not included in the repository to keep it lightweight.

---

## Project Structure

```
project_code/
├── src/
│   └── main.cu
├── results/
│   ├── logs.txt
│   ├── edge/
│   ├── equalized/
│   └── sharpen/
├── Makefile
├── run.sh
└── README.md
```

---

## How to Build

```bash
make
```

---

## How to Run

### Run full pipeline

```bash
./run.sh
```

---

### Run manually

```bash
./app <input_folder> <output_folder>
```

Example:

```bash
./app data/input/textures data/output
```

---

## Processing Modes

### 1. Edge Detection

* Converts RGB to grayscale
* Applies blur for noise reduction
* Uses Sobel operator for edge detection

### 2. Histogram Equalization

* Improves contrast of images
* Redistributes pixel intensity values

### 3. Sharpening

* Enhances image details using convolution filters

---

## Output

Processed images are saved in:

```
data/output/
 ├── edge/
 ├── equalized/
 └── sharpen/
```

---

## Sample Results

Sample output images are included in the `results/` directory:

* `results/edge/` → Edge detection outputs
* `results/equalized/` → Histogram equalized images
* `results/sharpen/` → Sharpened images

---

## Performance

Each processing operation includes GPU execution timing:

```
Time (edge): XX ms
Time (equalize): XX ms
Time (sharpen): XX ms
```

This demonstrates the efficiency of parallel GPU computation for large-scale image processing.

---

## Proof of Execution

* Execution logs: `results/logs.txt`
* Batch processing of 100+ images completed successfully
* Multiple GPU processing modes executed in a single run

---

## Challenges & Learnings

* Managing host ↔ device memory transfers
* Designing efficient CUDA kernels
* Implementing parallel image operations
* Handling performance variation across image sizes
* Building a configurable GPU processing pipeline

---

## Conclusion

This project demonstrates how CUDA-based GPU acceleration significantly improves performance for large-scale image processing tasks. The configurable pipeline design makes the system flexible, extensible, and suitable for real-world applications.
