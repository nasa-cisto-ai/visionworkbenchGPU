#include <iostream>
#include <chrono>
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
#include <cub/cub.cuh>
//#include <vw/Image/ImageView.h>

using namespace vw;
using namespace vw::stereo;

ImageView<float> compute_spatial_weight_image_cpu(int32 kern_width, int32 kern_height,
    float two_sigma_sqr) {
    int32 center_pix_x = kern_width/2;
    int32 center_pix_y = kern_height/2;

    float sum;
    sum = 0.0;
    ImageView<float> weight(kern_width, kern_height);
    for (int32 j = 0; j < kern_height; ++j) {
        for (int32 i = 0; i < kern_width; ++i ) {
            weight(i,j) = expf(-1*((i-center_pix_x)*(i-center_pix_x) +
                (j-center_pix_y)*(j-center_pix_y)) / two_sigma_sqr);
            sum += weight(i,j);
        }
    }

    weight /= sum;

    return weight;
}

__global__ void compute_spatial_weight_image(const int32 kern_width, const int32 kern_height,
    float* weight, const float two_sigma_sqr, const int32 center_pix_x, const int32 center_pix_y, float sum){
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIdx < kern_width && yIdx < kern_height){
        int idx = xIdx + yIdx * kern_width;
        weight[idx] = expf(-1*((xIdx-center_pix_x)*(xIdx-center_pix_x) +
            (yIdx-center_pix_y)*(yIdx-center_pix_y)) / two_sigma_sqr);
    }

    __syncthreads();
    
    // Not sure this is the best method, maybe a shared memory reduction?
    // atomicAdd was not working as expected
    if (xIdx == 0 && yIdx == 0) {
        float sum = 0.0;
        for (int y = 0; y < kern_height; ++y) {
            for (int x = 0; x < kern_width; ++x) {
                sum += weight[x + y * kern_width];
            }
        }

        for (int y = 0; y < kern_height; ++y) {
            for (int x = 0; x < kern_width; ++x) {
                weight[x + y * kern_width] /= sum;
            }
        }
    }
}

int main() {
    // For timing
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Spec spatial weight attributes
    int kern_width = 21;  // Specify your kernel width
    int kern_height = 21; // Specify your kernel height
    float two_sigma_sqr = 1.0f; // Specify your sigma value
    int32 center_pix_x = kern_width/2;
    int32 center_pix_y = kern_height/2;

    float* host_weight = new float[kern_width * kern_height];
    ImageView<float> hweight(kern_width, kern_height);
    float* device_weight;
    float sum = 0.0;
    cudaMalloc((void**)&device_weight, sizeof(float) * kern_width * kern_height);

    dim3 blockSize(16, 16);
    dim3 gridSize((kern_width + blockSize.x - 1) / blockSize.x, (kern_height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    auto t1 = high_resolution_clock::now();
    compute_spatial_weight_image<<<gridSize, blockSize>>>(kern_width, kern_height, device_weight, two_sigma_sqr, center_pix_x, center_pix_y, sum);
    auto t2 = high_resolution_clock::now();

    // Copy the result back to the CPU
    cudaMemcpy(hweight.data(), device_weight, sizeof(float) * kern_width * kern_height, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(device_weight);

    for (int y = 0; y < kern_height; y++) {
        for (int x = 0; x < kern_width; x++){
            std::cout << hweight(x, y) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Launch CPU implementation
    auto t3 = high_resolution_clock::now();
    ImageView<float> result = compute_spatial_weight_image_cpu(kern_width, kern_height, two_sigma_sqr);
    auto t4 = high_resolution_clock::now();

    for (int y = 0; y < kern_height; y++) {
        for (int x = 0; x < kern_width; x++){
            std::cout << result(x, y) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    auto ms_gpu_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_gpu_double = t2 - t1;
    std::cout << ms_gpu_int.count() << "ms GPU\n";
    std::cout << ms_gpu_double.count() << "ms GPU\n";

    auto ms_cpu_int = duration_cast<milliseconds>(t4 - t3);
    duration<double, std::milli> ms_cpu_double = t4 - t3;
    std::cout << ms_cpu_int.count() << "ms CPU\n";
    std::cout << ms_cpu_double.count() << "ms CPU\n";

    // m_data now contains the incremented values
    return 0;
}