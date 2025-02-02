#include "integral_image.h"

cv::Mat serial_version(const cv::Mat& image) {
    auto height = image.rows;
    auto width = image.cols;
    cv::Mat result(height, width, CV_32SC1);
    std::vector<int> col_sums(width, 0);

    // We iterate only once over the image and keep track of the current cumulative sum across rows for each column
    for (int i = 0; i < height; i++) {
        int current_sum = 0;
        for (int j = 0; j < width; j++) {
            col_sums[j] += image.at<uchar>(i, j);
            result.at<int>(i, j) = current_sum + col_sums[j];
            current_sum = result.at<int>(i, j);
        }
    }

    return result;
}
