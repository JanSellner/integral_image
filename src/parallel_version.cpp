#include "ParallelExecution.h"
#include "integral_image.h"

cv::Mat parallel_version(const cv::Mat& image) {
    auto height = image.rows;
    auto width = image.cols;
    cv::Mat result(height, width, CV_32SC1);

    // Cumsum across columns for each row in parallel
    ParallelExecution pe;
    pe.parallel_for(0, height - 1, [&](const size_t row) {
        int current_sum = 0;
        for (int col = 0; col < width; col++) {
            current_sum += image.at<uchar>(row, col);
            result.at<int>(row, col) = current_sum;
        }
    });

    // The second iteration is much more difficult to parallelize because of cache misses
    // A straightforward approach would be to simply parallelize the iterations over the columns (cf.
    // parallel_version2()) However, this has bad cache locality since every thread accesses the data from all rows. To
    // avoid this, we parallelize the operations across one row by letting each tread operate on a sequence of values (a
    // block)

    int n_blocks = pe.numbThreads;
    int block_size = std::ceil(static_cast<float>(width) / n_blocks);

    pe.parallel_for(0, n_blocks - 1, [&](const size_t b) {
        int block_start = b * block_size;
        int block_end = block_start + block_size;
        if (block_end > width) {
            block_end = width;
        }

        for (int row = 1; row < height; row++) {
            for (int col = block_start; col < block_end; col++) {
                result.at<int>(row, col) += result.at<int>(row - 1, col);
            }
        }
    });

    return result;
}

cv::Mat parallel_version2(const cv::Mat& image) {
    auto height = image.rows;
    auto width = image.cols;
    cv::Mat result(height, width, CV_32SC1);

    ParallelExecution pe;
    pe.parallel_for(0, height - 1, [&](const size_t row) {
        int current_sum = 0;
        for (int col = 0; col < width; col++) {
            current_sum += image.at<uchar>(row, col);
            result.at<int>(row, col) = current_sum;
        }
    });

    // Simple, but may be inefficient due to many cache misses
    pe.parallel_for(0, width - 1, [&](const size_t col) {
        int current_sum = 0;
        for (int row = 0; row < height; row++) {
            current_sum += result.at<int>(row, col);
            result.at<int>(row, col) = current_sum;
        }
    });

    return result;
}
