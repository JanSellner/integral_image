#include "integral_image.h"

bool _mat_equal(const cv::Mat& mat1, const cv::Mat& mat2) {
    return mat1.size() == mat2.size() && mat1.channels() == mat2.channels() && mat1.type() == mat2.type() &&
           std::equal(mat1.begin<int>(), mat1.end<int>(), mat2.begin<int>());
}

void test_cpu_functions(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    auto functions = {serial_version, parallel_version, parallel_version2, opencv_version, torch_version};

    for (size_t i = 0; i < inputs.size(); i++) {
        cv::Mat image = inputs[i];
        cv::Mat expected = targets[i];

        for (const auto& func : functions) {
            cv::Mat output = func(image);
            CV_Assert(_mat_equal(expected, output));
        }
    }
}

void test_gpu_functions(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    auto functions = {cuda_version};

    for (size_t i = 0; i < inputs.size(); i++) {
        CudaMat image(inputs[i]);
        cv::Mat expected = targets[i];
        CudaMat output(expected.rows, expected.cols, expected.type());

        for (const auto& func : functions) {
            func(image, output);
            CV_Assert(_mat_equal(expected, output.to_host()));
        }
    }
}

int main(int argc, char** argv) {
    auto inputs = std::vector<cv::Mat>{(cv::Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                       (cv::Mat_<uchar>(2, 3) << 1, 2, 3, 4, 5, 6),
                                       cv::Mat::ones(576, 768, CV_8UC1)};
    auto targets = std::vector<cv::Mat>{(cv::Mat_<int>(3, 3) << 1, 3, 6, 5, 12, 21, 12, 27, 45),
                                        (cv::Mat_<int>(2, 3) << 1, 3, 6, 5, 12, 21),
                                        opencv_version(inputs[2])};
    test_cpu_functions(inputs, targets);
    test_gpu_functions(inputs, targets);

    return 0;
}
