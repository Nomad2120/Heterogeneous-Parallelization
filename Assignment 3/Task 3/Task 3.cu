%%writefile task3.cu
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

__global__ void coalesced_read(const float *in, float *out, int n) { // коалесцированное чтение
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id потока
    if (tid < n) { // граница
        float x = in[tid]; // читаем подряд
        out[tid] = x * 2.0f; // пишем результат
    }
}

__global__ void noncoalesced_read(const float *in, float *out, int n, int stride) { // некоалесцированное чтение
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id потока
    if (tid < n) { // граница
        int idx = (tid * stride) % n; // скачок по памяти, но все потоки работают
        float x = in[idx]; // читаем не подряд
        out[tid] = x * 2.0f; // пишем результат (запись подряд)
    }
}

float time_coalesced(const float *d_in, float *d_out, int n, int block, int iters) { // замер coal
    int grid = (n + block - 1) / block; // grid
    cudaEvent_t s, e; // события
    cudaEventCreate(&s); cudaEventCreate(&e); // create

    coalesced_read<<<grid, block>>>(d_in, d_out, n); // прогрев
    cuda_check(cudaGetLastError(), "warmup coalesced"); // check
    cuda_check(cudaDeviceSynchronize(), "sync warmup coalesced"); // sync

    cudaEventRecord(s); // старт
    for (int i = 0; i < iters; i++) { // повторяем
        coalesced_read<<<grid, block>>>(d_in, d_out, n); // kernel
    }
    cuda_check(cudaGetLastError(), "kernel coalesced"); // check
    cuda_check(cudaDeviceSynchronize(), "sync coalesced"); // sync
    cudaEventRecord(e); cudaEventSynchronize(e); // стоп

    float ms = 0.0f; // время
    cudaEventElapsedTime(&ms, s, e); // мс
    cudaEventDestroy(s); cudaEventDestroy(e); // destroy
    return ms / iters; // среднее на один запуск
}

float time_noncoalesced(const float *d_in, float *d_out, int n, int block, int stride, int iters) { // замер non
    int grid = (n + block - 1) / block; // grid
    cudaEvent_t s, e; // события
    cudaEventCreate(&s); cudaEventCreate(&e); // create

    noncoalesced_read<<<grid, block>>>(d_in, d_out, n, stride); // прогрев
    cuda_check(cudaGetLastError(), "warmup noncoalesced"); // check
    cuda_check(cudaDeviceSynchronize(), "sync warmup noncoalesced"); // sync

    cudaEventRecord(s); // старт
    for (int i = 0; i < iters; i++) { // повторяем
        noncoalesced_read<<<grid, block>>>(d_in, d_out, n, stride); // kernel
    }
    cuda_check(cudaGetLastError(), "kernel noncoalesced"); // check
    cuda_check(cudaDeviceSynchronize(), "sync noncoalesced"); // sync
    cudaEventRecord(e); cudaEventSynchronize(e); // стоп

    float ms = 0.0f; // время
    cudaEventElapsedTime(&ms, s, e); // мс
    cudaEventDestroy(s); cudaEventDestroy(e); // destroy
    return ms / iters; // среднее на один запуск
}

int main() { // main
    const int N = 1000000; // размер
    const int block = 256; // block
    const int stride = 33; // шаг (не кратен 32, обычно хуже для доступа)
    const int iters = 30; // повторов для стабильности

    vector<float> h(N); // cpu
    for (int i = 0; i < N; i++) h[i] = (float)(i % 100); // данные

    float *d_in = nullptr; // вход на gpu
    float *d_out = nullptr; // выход на gpu
    cuda_check(cudaMalloc(&d_in, N * sizeof(float)), "malloc in"); // malloc
    cuda_check(cudaMalloc(&d_out, N * sizeof(float)), "malloc out"); // malloc
    cuda_check(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy in"); // копия

    float ms_coal = time_coalesced(d_in, d_out, N, block, iters); // замер
    float ms_non = time_noncoalesced(d_in, d_out, N, block, stride, iters); // замер

    cout << "размер: " << N << "\n"; // вывод
    cout << "block: " << block << "\n"; // вывод
    cout << "stride: " << stride << "\n"; // вывод
    cout << "коалесцировано: " << ms_coal << " мс\n"; // вывод
    cout << "некоалесцировано: " << ms_non << " мс\n"; // вывод

    cudaFree(d_in); // free
    cudaFree(d_out); // free
    return 0; // выход
}
