#include <torch/torch.h>
#include "integral_image.h"

cv::Mat torch_version(cv::Mat image) {
    auto height = image.rows;
    auto width = image.cols;
    cv::Mat result(height, width, CV_32SC1);

    // Create tensor views (no data is copied)
    auto image_tensor = torch::from_blob(image.data, {height, width}, torch::kByte);
    auto result_tensor = torch::from_blob(result.data, {height, width}, torch::kInt);

    // Store the output of the first cumsum directly in the results tensor
    torch::cumsum_out(result_tensor, image_tensor, 0);
    // The second iteration is performed in-place
    result_tensor.cumsum_(1);

    return result;
}
