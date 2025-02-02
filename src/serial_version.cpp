#include "integral_image.h"

cv::Mat serial_version(const cv::Mat& image) {
    auto height = image.rows;
    auto width = image.cols;
    cv::Mat result(height, width, CV_32SC1);
    std::vector<int> col_sums(width, 0);

    // We iterate only once over the image and keep track of the current cumulative sum across rows for each column
    for (int row = 0; row < height; row++) {
        int current_sum = 0;
        for (int col = 0; col < width; col++) {
            col_sums[col] += image.at<uchar>(row, col);
            result.at<int>(row, col) = current_sum + col_sums[col];
            current_sum = result.at<int>(row, col);
        }
    }

    return result;
}
