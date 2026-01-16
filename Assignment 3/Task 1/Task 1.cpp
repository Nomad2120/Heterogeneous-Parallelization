%%writefile task1.cu
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

__global__ void mul_global(float *a, float k, int n) { // умножение через global
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id
    if (tid < n) a[tid] = a[tid] * k; // умножаем
}

__global__ void mul_shared(float *a, float k, int n) { // умножение через shared
    extern __shared__ float sh[]; // shared буфер
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id
    int local = threadIdx.x; // локальный id
    if (tid < n) sh[local] = a[tid]; // грузим в shared
    __syncthreads(); // синхронизация
    if (tid < n) sh[local] = sh[local] * k; // умножаем в shared
    __syncthreads(); // синхронизация
    if (tid < n) a[tid] = sh[local]; // пишем назад
}

int main() { // main
    const int N = 1000000; // размер
    const float k = 3.0f; // множитель
    const int block = 256; // block size
    const int grid = (N + block - 1) / block; // grid size

    vector<float> h(N); // массив на cpu
    for (int i = 0; i < N; i++) h[i] = (float)(i % 100); // простые данные

    float *d = nullptr; // массив на gpu
    cuda_check(cudaMalloc(&d, N * sizeof(float)), "cudaMalloc"); // malloc
    cuda_check(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D"); // копия

    cudaEvent_t s1, e1, s2, e2; // события
    cudaEventCreate(&s1); cudaEventCreate(&e1); // для global
    cudaEventCreate(&s2); cudaEventCreate(&e2); // для shared

    cudaEventRecord(s1); // старт global
    mul_global<<<grid, block>>>(d, k, N); // запуск
    cuda_check(cudaGetLastError(), "kernel global"); // check
    cuda_check(cudaDeviceSynchronize(), "sync global"); // ждем
    cudaEventRecord(e1); cudaEventSynchronize(e1); // стоп
    float ms_global = 0.0f; cudaEventElapsedTime(&ms_global, s1, e1); // время

    cuda_check(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice), "reset H2D"); // сброс данных

    size_t sh_bytes = block * sizeof(float); // shared bytes
    cudaEventRecord(s2); // старт shared
    mul_shared<<<grid, block, sh_bytes>>>(d, k, N); // запуск
    cuda_check(cudaGetLastError(), "kernel shared"); // check
    cuda_check(cudaDeviceSynchronize(), "sync shared"); // ждем
    cudaEventRecord(e2); cudaEventSynchronize(e2); // стоп
    float ms_shared = 0.0f; cudaEventElapsedTime(&ms_shared, s2, e2); // время

    cout << "размер: " << N << "\n"; // вывод
    cout << "block: " << block << "\n"; // вывод
    cout << "global память: " << ms_global << " мс\n"; // вывод
    cout << "shared память: " << ms_shared << " мс\n"; // вывод

    cudaFree(d); // free
    return 0; // выход
}
