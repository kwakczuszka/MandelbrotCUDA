#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

__global__ void mandelbrot_kernel(uint32_t* pixels, int width, int height,
    double centerX, double centerY, double scale, int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width*2 || y >= height*2) return;

    double zx = (x - width) * scale + centerX;
    double zy = (y - height) * scale + centerY;
    double cx = zx, cy = zy;

    int i = 0;
    double zx_sq = zx * zx, zy_sq = zy * zy;    // saves 2 mults per iteration
    while (zx_sq + zy_sq < 4.0 && i < maxIter) {
        double tmp = zx_sq - zy_sq + cx;
        zy = std::fma(2.0, zx * zy, cy);
        zx = tmp;
        zx_sq = zx * zx;
        zy_sq = zy * zy;
        ++i;
    }

    uint32_t color = (i == maxIter) ? 0x000000FF : (0xFF << 24) 
                                                    | ((i * 9) % 255 << 16) 
                                                    | ((i * 5) % 255 << 8) 
                                                    | (i * 2) % 255;
    pixels[y * width*2 + x] = color;
}

void launch_mandelbrot(uint32_t * devBuffer, int width, int height,
    double centerX, double centerY, double scale, int maxIter) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    //passing width/2 and height/2 to avoid unnecessary divisions later
    mandelbrot_kernel <<<blocks, threads >>> (devBuffer, width/2, height/2, centerX, centerY, scale, maxIter);  
    cudaDeviceSynchronize();
}
