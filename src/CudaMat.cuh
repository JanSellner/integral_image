#pragma once

#include <opencv2/opencv.hpp>

/**
 * @class CudaMat
 * @brief A small helper class to manage matrix data on cuda devices. Data passed to this class will be copied to the
 * device memory and lives for the lifetime of this class.
 */
class CudaMat {
   public:
    /**
     * @brief Transfers the given image data to device memory.
     *
     * @param mat The matrix with the image data to be transferred to the GPU.
     */
    CudaMat(const cv::Mat& mat);

    /**
     * @brief Allocates memory on the device for a matrix with the given dimensions and type.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param type The OpenCV type of the matrix.
     */
    CudaMat(int rows, int cols, int type);

    /**
     * @brief Frees the memory allocated on the device.
     */
    ~CudaMat();

    /**
     * @brief Retrieve a pointer to the device data.
     *
     * @tparam T The type of the image data.
     *
     * @return T* A pointer to the device data.
     */
    template <typename T>
    T* ptr() {
        return static_cast<T*>(this->device_data);
    }
    /**
     * @copydoc CudaMat::ptr()
     */
    template <typename T>
    const T* ptr() const {
        return static_cast<T*>(this->device_data);
    }

    /**
     * @brief Transfers the data from the device to the host and returns an image with the data.
     *
     * @return cv::Mat The image with the data from the device.
     */
    cv::Mat to_host();

    //! The number of rows in the matrix.
    int rows;
    //! The number of columns in the matrix.
    int cols;
    //! The OpenCV type of the matrix.
    int type;

   private:
    void* device_data;
};
