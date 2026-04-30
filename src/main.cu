#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <string>

namespace fs = std::filesystem;

// ===================== CUDA KERNELS =====================

// Grayscale
__global__ void rgb_to_gray(unsigned char* input, unsigned char* gray, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = (y * w + x) * 3;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        gray[y * w + x] = 0.299f*r + 0.587f*g + 0.114f*b;
    }
}

// Blur
__global__ void blur_kernel(unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < w-1 && y < h-1) {
        int sum = 0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                sum += in[(y+dy)*w + (x+dx)];
            }
        }
        out[y*w + x] = sum / 9;
    }
}

// Sobel
__global__ void sobel_kernel(unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < w-1 && y < h-1) {
        int gx = -in[(y-1)*w + (x-1)] - 2*in[y*w + (x-1)] - in[(y+1)*w + (x-1)]
                 + in[(y-1)*w + (x+1)] + 2*in[y*w + (x+1)] + in[(y+1)*w + (x+1)];

        int gy = -in[(y-1)*w + (x-1)] - 2*in[(y-1)*w + x] - in[(y-1)*w + (x+1)]
                 + in[(y+1)*w + (x-1)] + 2*in[(y+1)*w + x] + in[(y+1)*w + (x+1)];

        int mag = min(255, abs(gx) + abs(gy));
        out[y*w + x] = mag;
    }
}

// Histogram kernel
__global__ void histogramKernel(unsigned char* img, int* hist, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&hist[img[idx]], 1);
    }
}

// Equalization kernel
__global__ void equalizeKernel(unsigned char* img, unsigned char* out, float* cdf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (unsigned char)(255 * cdf[img[idx]]);
    }
}

// Sharpen kernel
__global__ void sharpenKernel(unsigned char* img, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < w-1 && y < h-1) {
        int idx = y * w + x;

        int val = 5 * img[idx]
            - img[idx - 1]
            - img[idx + 1]
            - img[idx - w]
            - img[idx + w];

        out[idx] = min(max(val, 0), 255);
    }
}

// ===================== PROCESS FUNCTION =====================

void process_image(const std::string& input_path,
                   const std::string& output_path,
                   const std::string& mode) {

    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Failed: " << input_path << std::endl;
        return;
    }

    int w = img.cols;
    int h = img.rows;
    int size = w * h;

    unsigned char *d_input, *d_gray, *d_temp, *d_out;

    cudaMalloc(&d_input, w*h*3);
    cudaMalloc(&d_gray, size);
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_input, img.data, w*h*3, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((w+15)/16, (h+15)/16);

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    rgb_to_gray<<<grid, block>>>(d_input, d_gray, w, h);

    if (mode == "edge") {
        blur_kernel<<<grid, block>>>(d_gray, d_temp, w, h);
        sobel_kernel<<<grid, block>>>(d_temp, d_out, w, h);
    }

    else if (mode == "sharpen") {
        sharpenKernel<<<grid, block>>>(d_gray, d_out, w, h);
    }

    else if (mode == "equalize") {

        int* d_hist;
        cudaMalloc(&d_hist, 256 * sizeof(int));
        cudaMemset(d_hist, 0, 256 * sizeof(int));

        histogramKernel<<<(size+255)/256, 256>>>(d_gray, d_hist, size);

        int h_hist[256];
        cudaMemcpy(h_hist, d_hist, 256*sizeof(int), cudaMemcpyDeviceToHost);

        float cdf[256];
        cdf[0] = h_hist[0];

        for (int i = 1; i < 256; i++)
            cdf[i] = cdf[i-1] + h_hist[i];

        for (int i = 0; i < 256; i++)
            cdf[i] /= cdf[255];

        float* d_cdf;
        cudaMalloc(&d_cdf, 256*sizeof(float));
        cudaMemcpy(d_cdf, cdf, 256*sizeof(float), cudaMemcpyHostToDevice);

        equalizeKernel<<<(size+255)/256, 256>>>(d_gray, d_out, d_cdf, size);

        cudaFree(d_hist);
        cudaFree(d_cdf);
    }

    // Timing end
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Time (" << mode << "): " << time << " ms\n";

    cv::Mat result(h, w, CV_8UC1);
    cudaMemcpy(result.data, d_out, size, cudaMemcpyDeviceToHost);

    cv::imwrite(output_path, result);

    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_temp);
    cudaFree(d_out);
}

// ===================== MAIN =====================

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cout << "Usage: ./app <input_folder> <output_folder> --mode=edge\n";
        return -1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];

    std::string mode = "edge"; // default

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--mode=") == 0) {
            mode = arg.substr(7);
        }
    }

    int count = 0;

    for (const auto& entry : fs::directory_iterator(input_folder)) {

        std::string input_path = entry.path().string();
        std::string filename = entry.path().filename().string();

        std::string output_path = output_folder + "/" + mode + "_" + filename;

        process_image(input_path, output_path, mode);

        count++;
        std::cout << "Processed: " << filename << std::endl;
    }

    std::cout << "Total images processed: " << count << std::endl;

    return 0;
}