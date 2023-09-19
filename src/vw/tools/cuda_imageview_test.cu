#include <iostream>
#include <cuda_runtime.h>

#include <vw/Math/Vector.h>
#include <vw/Math/Matrix.h>
#include <vw/Image/PixelTypeInfo.h>
#include <vw/Image/PixelMath.h>
#include <vw/Math/BBox.h>
#include <vw/Core/Debugging.h>
#include <vw/Math/Geometry.h>
#include <vw/Math/RANSAC.h>
#include <vw/Math/Vector.h>
#include <vw/Image/Algorithms.h>
#include <vw/Image/EdgeExtension.h>
#include <vw/Image/Manipulation.h>
#include <vw/Image/MaskViews.h>
#include <vw/Image/Transform.h>
#include <vw/InterestPoint/InterestData.h>
#include <vw/InterestPoint/Matcher.h>
#include <vw/Stereo/CorrelationView.h>
#include <vw/Stereo/CostFunctions.h>
#include <vw/Stereo/PreFilter.h>
#include <vw/Stereo/DisparityMap.h>
#include <vw/Stereo/Correlate.h>
//#include <vw/Image/ImageView.h>

using namespace vw;
using namespace vw::stereo;

__global__ void addOneToImageView(const ImageView<PixelGray<float>> input,  const int32 cols, const int32 rows, ImageView<PixelGray<float>> result){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows) {
        int idx = row * cols + col;
    }
}

int main() {
    // just to test we can do this
    int lsize = 100;
    ImageView<PixelGray<float>> dummy_data(4*lsize, 4*lsize);

    for (int col = 0; col < dummy_data.cols(); col++){
        for (int row = 0; row < dummy_data.rows(); row++){
            dummy_data(col, row) = col%2 + 2*(row%5); // some values
        }
    }

    for (int col = 0; col < dummy_data.cols(); col++){
        for (int row = 0; row < dummy_data.rows(); row++){
            dummy_data(col, row) = col%2 + 2*(row%5); // some values
        }
    }

    PixelGray<float> pixelSample = dummy_data(0, 0);
    float sc = static_cast<float>(pixelSample);

    std::cout << sc << std::endl;


    // Define variables for GPU memory
    PixelGray<float>* gpu_dummy_data = nullptr;
    PixelGray<float>* gpu_result_data = nullptr;
    size_t data_size = dummy_data.cols() * dummy_data.rows() * sizeof(PixelGray<float>);

    // Allocate GPU memory
    cudaMalloc((void**)&gpu_dummy_data, data_size);
    cudaMalloc((void**)&gpu_result_data, data_size);

    // Check for allocation errors
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed." << std::endl;
        return 1;
    }

    // Copy data from CPU to GPU
    cudaMemcpy(gpu_dummy_data, dummy_data.data(), data_size, cudaMemcpyHostToDevice);

    // Check for copy errors
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA memory copy failed." << std::endl;
        return 1;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((dummy_data.cols() + blockDim.x - 1) / blockDim.x, (dummy_data.rows() + blockDim.y - 1) / blockDim.y);

    // Perform GPU processing here if needed

    // Free GPU memory when done
    cudaFree(gpu_dummy_data);

    return 0;
}
