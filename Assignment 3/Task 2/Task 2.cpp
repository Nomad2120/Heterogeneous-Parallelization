%%writefile task2.cu
#include <iostream> // вывод
#include <vector> // вектор
#include <cuda_runtime.h> // cuda

using namespace std; // чтоб не писать std::

void cuda_check(cudaError_t err, const char* place) { // проверка cuda
    if (err != cudaSuccess) { // если ошибка
        cout << "ошибка cuda в " << place << ": " << cudaGetErrorString(err) << "\n"; // вывод
        exit(1); // выход
    }
}

__global__ void add_arrays(const float *a, const float *b, float *c, int n) { // сложение
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id
    if (tid < n) c[tid] = a[tid] + b[tid]; // сумма
}

float run_with_block(int block, const float *d_a, const float *d_b, float *d_c, int n) { // запуск с block
    int grid = (n + block - 1) / block; // grid
    cudaEvent_t s, e; // события
    cudaEventCreate(&s); cudaEventCreate(&e); // create
    cudaEventRecord(s); // старт
    add_arrays<<<grid, block>>>(d_a, d_b, d_c, n); // kernel
    cuda_check(cudaGetLastError(), "kernel"); // check
    cuda_check(cudaDeviceSynchronize(), "sync"); // sync
    cudaEventRecord(e); cudaEventSynchronize(e); // стоп
    float ms = 0.0f; cudaEventElapsedTime(&ms, s, e); // время
    cudaEventDestroy(s); cudaEventDestroy(e); // destroy
    return ms; // вернуть
}

int main() { // main
    const int N = 1000000; // размер
    vector<float> h_a(N), h_b(N); // cpu массивы
    for (int i = 0; i < N; i++) { h_a[i] = (float)(i % 50); h_b[i] = (float)(i % 80); } // данные

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr; // gpu указатели
    cuda_check(cudaMalloc(&d_a, N * sizeof(float)), "malloc a"); // malloc
    cuda_check(cudaMalloc(&d_b, N * sizeof(float)), "malloc b"); // malloc
    cuda_check(cudaMalloc(&d_c, N * sizeof(float)), "malloc c"); // malloc
    cuda_check(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy a"); // копия
    cuda_check(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy b"); // копия

    int blocksizes[3] = {128, 256, 512}; // три размера блока
    float times[3]; // времена

    for (int i = 0; i < 3; i++) { // цикл
        times[i] = run_with_block(blocksizes[i], d_a, d_b, d_c, N); // замер
    }

    cout << "размер: " << N << "\n"; // вывод
    for (int i = 0; i < 3; i++) { // вывод
        cout << "block: " << blocksizes[i] << " время: " << times[i] << " мс\n"; // вывод
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); // free
    return 0; // выход
}
