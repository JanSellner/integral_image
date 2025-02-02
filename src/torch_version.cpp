#include <torch/torch.h>
#include "integral_image.h"

cv::Mat torch_version(const cv::Mat& image) {
    auto height = image.rows;
    auto width = image.cols;
    cv::Mat result(height, width, CV_32SC1);

    // Create tensor views (no data is copied)
    auto image_view = torch::from_blob(image.data, {height, width}, torch::kByte);
    auto result_view = torch::from_blob(result.data, {height, width}, torch::kInt);

    // Store the output of the first cumsum directly in the results tensor
    torch::cumsum_out(result_view, image_view, 0);
    // The second iteration is performed in-place
    result_view.cumsum_(1);

    return result;
}
