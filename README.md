# Configurable GPU Image Processing Pipeline using CUDA

## Overview

This project implements a configurable GPU-accelerated image processing pipeline using CUDA. It processes over 100 images from the USC SIPI dataset and applies multiple transformations using custom CUDA kernels.

Unlike a fixed pipeline, this system supports multiple processing modes through command-line arguments, allowing dynamic selection of different GPU-based operations such as edge detection, histogram equalization, and sharpening.

---

## Features

* Batch processing of 100+ images
* GPU acceleration using CUDA kernels
* Multiple configurable processing modes:

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

---

## Project Structure

```
cuda_image_project/
├── data/
│   ├── input/
│   └── output/
│       ├── edge/
│       ├── equalized/
│       └── sharpen/
├── src/
│   └── main.cu
├── results/
│   ├── logs.txt
│   ├── before/
│   └── after/
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

### Run full pipeline (all modes)

```bash
./run.sh
```

### Run specific mode manually

```bash
./app data/input/textures data/output --mode=edge
./app data/input/textures data/output --mode=equalize
./app data/input/textures data/output --mode=sharpen
```

---

## Processing Modes

### 1. Edge Detection

* Converts RGB to grayscale
* Applies blur for noise reduction
* Uses Sobel operator for edge detection

### 2. Histogram Equalization

* Computes image histogram on GPU
* Normalizes intensity distribution
* Enhances contrast

### 3. Sharpening

* Applies convolution-based sharpening filter
* Enhances fine details in images

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

### Example (Edge Detection)

**Input Image:**

![Before](results/before/house.png)

**Output:**

![After](results/after/house.png)

---

## Performance

Each image processing operation includes GPU execution timing:

```
Time (edge): XX ms
Time (equalize): XX ms
Time (sharpen): XX ms
```

This demonstrates the efficiency of parallel GPU computation for large-scale image processing.

---

## Proof of Execution

* Execution logs: `results/logs.txt`
* Batch processing of 100+ images across multiple modes
* Output images generated for each processing type

---

## Challenges & Learnings

* Managing memory transfers between CPU and GPU
* Designing efficient CUDA kernels for pixel-wise operations
* Implementing histogram equalization using parallel computation
* Structuring a configurable pipeline using CLI arguments
* Understanding performance implications of GPU parallelism

---

## Conclusion

This project demonstrates how GPU acceleration using CUDA can significantly enhance performance for large-scale image processing tasks. By introducing configurable modes and multiple GPU kernels, the system evolves from a fixed pipeline to a flexible and extensible image processing framework.
