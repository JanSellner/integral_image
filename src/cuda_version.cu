#include "integral_image.h"

__global__ void integral_image_rows(const uchar* image, int* result, int height, int width) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= height) {
        return;
    }

    // Cumsum across columns for each row in parallel
    int current_sum = 0;
    for (int col = 0; col < width; col++) {
        current_sum += image[row * width + col];
        result[row * width + col] = current_sum;
    }
}

__global__ void integral_image_cols(int* result, int height, int width) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= width) {
        return;
    }

    // Cumsum across rows for each column in parallel
    int current_sum = 0;
    for (int row = 0; row < height; row++) {
        current_sum += result[row * width + col];
        result[row * width + col] = current_sum;
    }
}

void cuda_version(const CudaMat& image, CudaMat& result) {
    auto height = image.rows;
    auto width = image.cols;

    // Similar to parallel_naive_version but with one thread for each row/column
    int threads = 64;
    int blocks = (height + threads - 1) / threads;
    integral_image_rows<<<blocks, threads>>>(image.ptr<uchar>(), result.ptr<int>(), height, width);

    blocks = (width + threads - 1) / threads;
    integral_image_cols<<<blocks, threads>>>(result.ptr<int>(), height, width);

    // For time measuring purposes
    cudaDeviceSynchronize();
}
