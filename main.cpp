#include <SDL.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>

extern void launch_mandelbrot(uint32_t* devBuffer, int width, int height,
    double centerX, double centerY, double scale, int maxIter);

const int WIDTH = 1600;
const int HEIGHT = 900;
const int MAX_ITER = 1000;

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL init failed!\n";
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("CUDA Mandelbrot", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_XRGB8888, SDL_TEXTUREACCESS_STREAMING,
        WIDTH, HEIGHT);

    uint32_t* pixels_host = new uint32_t[WIDTH * HEIGHT];
    uint32_t* pixels_dev;
    cudaMalloc(&pixels_dev, WIDTH * HEIGHT * sizeof(uint32_t));

    double scale = 0.005;
    double centerX = -0.5;
    double centerY = 0.0;

    bool running = true;
    SDL_Event e;
    while (running) {
        // Input handling
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN) {
                double moveFactor = 50 * scale;  // move 50 pixels
                switch (e.key.keysym.sym) {
                case SDLK_w: centerY -= moveFactor; break;
                case SDLK_s: centerY += moveFactor; break;
                case SDLK_a: centerX -= moveFactor; break;
                case SDLK_d: centerX += moveFactor; break;
                case SDLK_q: scale *= 0.8; break;
                case SDLK_e: scale /= 0.8; break;
                }
            }
        }

        launch_mandelbrot(pixels_dev, WIDTH, HEIGHT, centerX, centerY, scale, MAX_ITER);

        cudaMemcpy(pixels_host, pixels_dev, WIDTH * HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        SDL_UpdateTexture(texture, nullptr, pixels_host, WIDTH * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    cudaFree(pixels_dev);
    delete[] pixels_host;
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
