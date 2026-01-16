%%writefile task4.cu
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

float run_cfg(int block, int grid, const float *d_a, const float *d_b, float *d_c, int n) { // запуск
    cudaEvent_t s, e; // события
    cudaEventCreate(&s); cudaEventCreate(&e); // create
    cudaEventRecord(s); // старт
    add_arrays<<<grid, block>>>(d_a, d_b, d_c, n); // kernel
    cuda_check(cudaGetLastError(), "kernel"); // check
    cuda_check(cudaDeviceSynchronize(), "sync"); // sync
    cudaEventRecord(e); cudaEventSynchronize(e); // стоп
    float ms = 0.0f; // мс
    cudaEventElapsedTime(&ms, s, e); // время
    cudaEventDestroy(s); cudaEventDestroy(e); // destroy
    return ms; // вернуть
}

int main() { // main
    const int N = 1000000; // размер
    vector<float> h_a(N), h_b(N); // cpu
    for (int i = 0; i < N; i++) { // цикл
        h_a[i] = (float)(i % 10); // данные
        h_b[i] = (float)(i % 20); // данные
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr; // gpu
    cuda_check(cudaMalloc(&d_a, N * sizeof(float)), "malloc a"); // malloc
    cuda_check(cudaMalloc(&d_b, N * sizeof(float)), "malloc b"); // malloc
    cuda_check(cudaMalloc(&d_c, N * sizeof(float)), "malloc c"); // malloc
    cuda_check(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy a"); // копия
    cuda_check(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy b"); // копия

    int block_bad = 32; // плохой block (мало потоков)
    int grid_bad = (N + block_bad - 1) / block_bad; // важно: покрываем весь массив
    float ms_bad = run_cfg(block_bad, grid_bad, d_a, d_b, d_c, N); // время

    int block_good = 256; // хороший block
    int grid_good = (N + block_good - 1) / block_good; // покрываем весь массив
    float ms_good = run_cfg(block_good, grid_good, d_a, d_b, d_c, N); // время

    cout << "размер: " << N << "\n"; // вывод
    cout << "плохая конфигурация: block=" << block_bad << " grid=" << grid_bad << " время=" << ms_bad << " мс\n"; // вывод
    cout << "хорошая конфигурация: block=" << block_good << " grid=" << grid_good << " время=" << ms_good << " мс\n"; // вывод
    cout << "ускорение (bad/good): " << (ms_bad / (ms_good + 1e-6f)) << "\n"; // вывод

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); // free
    return 0; // выход
}
