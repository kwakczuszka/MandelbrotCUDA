#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

__global__ void mandelbrot_kernel(uint32_t* pixels, int width, int height,
    double centerX, double centerY, double scale, int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    double zx = (x - width / 2.0) * scale + centerX;
    double zy = (y - height / 2.0) * scale + centerY;
    double cx = zx, cy = zy;

    int i = 0;
    while (zx * zx + zy * zy < 4.0 && i < maxIter) {
        double tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        ++i;
    }

    uint32_t color = (i == maxIter) ? 0x000000FF : (0xFF << 24) 
                                                    | ((i * 9) % 255 << 16) 
                                                    | ((i * 5) % 255 << 8) 
                                                    | (i * 2) % 255;
    pixels[y * width + x] = color;
}

void launch_mandelbrot(uint32_t * devBuffer, int width, int height,
    double centerX, double centerY, double scale, int maxIter) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    mandelbrot_kernel <<<blocks, threads >>> (devBuffer, width, height, centerX, centerY, scale, maxIter);
    cudaDeviceSynchronize();
}
