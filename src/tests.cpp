#include "integral_image.h"

bool _mat_equal(const cv::Mat& mat1, const cv::Mat& mat2) {
    return mat1.size() == mat2.size() && mat1.channels() == mat2.channels() && mat1.type() == mat2.type() &&
           std::equal(mat1.begin<int>(), mat1.end<int>(), mat2.begin<int>());
}

void test_square(const std::vector<std::function<cv::Mat(cv::Mat)>>& functions) {
    cv::Mat image = (cv::Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    cv::Mat expected = (cv::Mat_<int>(3, 3) << 1, 3, 6, 5, 12, 21, 12, 27, 45);

    for (const auto& func : functions) {
        cv::Mat output = func(image);
        CV_Assert(_mat_equal(expected, output));
    }
}

void test_rectangle(const std::vector<std::function<cv::Mat(cv::Mat)>>& functions) {
    cv::Mat image = (cv::Mat_<uchar>(2, 3) << 1, 2, 3, 4, 5, 6);
    cv::Mat expected = (cv::Mat_<int>(2, 3) << 1, 3, 6, 5, 12, 21);

    for (const auto& func : functions) {
        cv::Mat output = func(image);
        CV_Assert(_mat_equal(expected, output));
    }
}

void test_image(const std::vector<std::function<cv::Mat(cv::Mat)>>& functions) {
    cv::Mat image = cv::Mat::ones(576, 768, CV_8UC1);
    cv::Mat expected = opencv_version(image);

    for (const auto& func : functions) {
        cv::Mat output = func(image);
        CV_Assert(_mat_equal(expected, output));
    }
}

int main(int argc, char** argv) {
    std::vector<std::function<cv::Mat(cv::Mat)>> functions = {serial_version,
                                                              parallel_version,
                                                              parallel_version2,
                                                              opencv_version,
                                                              torch_version};

    test_square(functions);
    test_rectangle(functions);
    test_image(functions);

    return 0;
}
