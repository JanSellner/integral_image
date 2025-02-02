#include "integral_image.h"

__global__ void integral_image_rows(const uchar* image, int* result, int height, int width) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= height) {
        return;
    }

    // Cumsum across columns for each row in parallel
    int current_sum = 0;
    for (int j = 0; j < width; j++) {
        current_sum += image[i * width + j];
        result[i * width + j] = current_sum;
    }
}

__global__ void integral_image_cols(const int* image, int* result, int height, int width) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= width) {
        return;
    }

    // Cumsum across rows for each column in parallel
    int current_sum = 0;
    for (int i = 0; i < height; i++) {
        current_sum += image[i * width + j];
        result[i * width + j] = current_sum;
    }
}

void cuda_version(const CudaMat& image, CudaMat& result) {
    auto height = image.rows;
    auto width = image.cols;

    // Similar to parallel_version2 but with one thread for each row/column
    int threads = 64;
    int blocks = (height + threads - 1) / threads;
    integral_image_rows<<<blocks, threads>>>(image.ptr<uchar>(), result.ptr<int>(), height, width);

    blocks = (width + threads - 1) / threads;
    integral_image_cols<<<blocks, threads>>>(result.ptr<int>(), result.ptr<int>(), height, width);

    // For time measuring purposes
    cudaDeviceSynchronize();
}
