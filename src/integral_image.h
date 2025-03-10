#pragma once

#include <opencv2/opencv.hpp>
#include "CudaMat.cuh"

/**
 * @brief Computes the integral image of the input image using OpenCV.
 *
 * The integral image is a representation in which the value at each pixel
 * is the sum of all the pixels above and to the left of it, inclusive.
 *
 * @param image The single-channel uint8 input image for which the integral image should be computed.
 * @return The integral image with the same dimensions as the input image but with int32 dtype.
 */
inline cv::Mat opencv_version(const cv::Mat& image) {
    cv::Mat result;
    cv::integral(image, result);

    // The OpenCV version includes an additional first row and column with zeros
    result = result(cv::Range(1, result.rows), cv::Range(1, result.cols));
    return result;
}

/**
 * @brief Computes the integral image of the input image using a serial algorithm.
 *
 * @copydetails opencv_version()
 */
cv::Mat serial_version(const cv::Mat& image);

/**
 * @brief Computes the integral image of the input image using a parallel algorithm.
 *
 * The parallelization is done by computing the cumulative row and column sums in parallel for multiple rows/columns at
 * once.
 *
 * @copydetails opencv_version()
 */
cv::Mat parallel_cache_version(const cv::Mat& image);

/**
 * @brief Computes the integral image of the input image using a parallel algorithm.
 *
 * Compared to parallel_cache_version(), this version parallelizes the second iteration so that first the sum over over
 * a row is computed instead of first finishing the sums for one row. This has worse cache locality.
 *
 * @copydetails opencv_version()
 */
cv::Mat parallel_naive_version(const cv::Mat& image);

/**
 * @brief Computes the integral image of the input image using using torch.
 *
 * Computing the integral image in PyTorch can be achieved via `x.cumsum(0).cumsum(1)`.
 *
 * @copydetails opencv_version()
 */
cv::Mat torch_version(const cv::Mat& image);

/**
 * @brief Computes the integral image of the input image using using cuda.
 *
 * The data passed to this function must already be on the GPU.
 *
 * @param image The single-channel uint8 input image for which the integral image should be computed.
 * @param result The output where the integral image should be stored. Must have same dimensions as the input image but
 * with int32 dtype.
 */
void cuda_version(const CudaMat& image, CudaMat& result);
