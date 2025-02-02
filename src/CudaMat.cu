#include "CudaMat.cuh"

CudaMat::CudaMat(const cv::Mat& mat) {
    cudaMalloc((void**)&this->device_data, mat.rows * mat.cols * CV_ELEM_SIZE(mat.type()));
    cudaMemcpy(this->device_data, mat.data, mat.rows * mat.cols * CV_ELEM_SIZE(mat.type()), cudaMemcpyHostToDevice);
    this->rows = mat.rows;
    this->cols = mat.cols;
    this->type = mat.type();
}

CudaMat::CudaMat(int rows, int cols, int type) {
    cudaMalloc((void**)&this->device_data, rows * cols * CV_ELEM_SIZE(type));
    this->rows = rows;
    this->cols = cols;
    this->type = type;
}

CudaMat::~CudaMat() {
    cudaFree(this->device_data);
}

cv::Mat CudaMat::to_host() {
    cv::Mat mat(this->rows, this->cols, this->type);
    cudaMemcpy(mat.data, this->device_data, this->rows * this->cols * CV_ELEM_SIZE(this->type), cudaMemcpyDeviceToHost);
    return mat;
}
