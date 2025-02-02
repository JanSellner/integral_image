#include <chrono>
#include <iostream>
#include "integral_image.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " SCALE serial|parallel|parallel2|opencv|torch" << std::endl;
        return 1;
    }

    cv::Mat image = cv::Mat::ones(576, 768, CV_8UC1);

    CV_Assert(image.isContinuous());
    CV_Assert(image.depth() == CV_8U);

    float scaling_factor = std::stof(argv[1]);
    int n_warmup = 3;
    int n_repeats = 10;
    cv::Mat result;

    cv::resize(image, image, cv::Size(), scaling_factor, scaling_factor);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    if (std::string(argv[2]) == "cuda") {
        CudaMat image_cuda(image);
        CudaMat result(image.rows, image.cols, CV_32SC1);

        // Warmup
        for (int i = 0; i < n_warmup; i++) {
            cuda_version(image_cuda, result);
        }

        begin = std::chrono::steady_clock::now();
        for (int i = 0; i < n_repeats; i++) {
            cuda_version(image_cuda, result);
        }
        end = std::chrono::steady_clock::now();
    } else {
        std::function<cv::Mat(const cv::Mat&)> func;
        if (std::string(argv[2]) == "serial") {
            func = serial_version;
        } else if (std::string(argv[2]) == "parallel") {
            func = parallel_version;
        } else if (std::string(argv[2]) == "parallel2") {
            func = parallel_version2;
        } else if (std::string(argv[2]) == "opencv") {
            func = opencv_version;
        } else if (std::string(argv[2]) == "torch") {
            func = torch_version;
        } else {
            std::cerr << "Invalid run type: " << argv[2] << std::endl;
            exit(2);
        }

        // Warmup
        for (int i = 0; i < n_warmup; i++) {
            result = func(image);
        }

        begin = std::chrono::steady_clock::now();
        for (int i = 0; i < n_repeats; i++) {
            result = func(image);
        }
        end = std::chrono::steady_clock::now();
    }

    // Pass the results back to the caller
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() /
                     static_cast<float>(n_repeats);

    return 0;
}
