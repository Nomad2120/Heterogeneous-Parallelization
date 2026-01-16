%%writefile bonus_tasks.cu
#include <iostream> // вывод
#include <vector> // вектор
#include <cuda_runtime.h> // cuda
#include <chrono> // cpu время

using namespace std; // чтоб не писать std::

void cuda_check(cudaError_t err, const char* place) { // проверка cuda
    if (err != cudaSuccess) { // если ошибка
        cout << "ошибка cuda в " << place << ": " << cudaGetErrorString(err) << "\n"; // вывод
        exit(1); // выход
    }
}

struct QueueMPMC { // простая mpmc очередь
    int *data; // буфер
    int *head; // head
    int *tail; // tail
    int cap; // емкость

    __device__ void init(int *buf, int *h, int *t, int c) { // init
        data = buf; head = h; tail = t; cap = c; // присваиваем
        if (blockIdx.x == 0 && threadIdx.x == 0) { *head = 0; *tail = 0; } // сброс
        __syncthreads(); // барьер
    }

    __device__ bool enqueue(int v) { // enqueue
        int pos = atomicAdd(tail, 1); // слот
        if (pos < cap) { data[pos] = v; return true; } // кладем
        atomicSub(tail, 1); return false; // откат
    }

    __device__ bool dequeue(int *v) { // dequeue
        int pos = atomicAdd(head, 1); // слот
        int tnow = atomicAdd(tail, 0); // tail
        if (pos < tnow) { *v = data[pos]; return true; } // берем
        atomicSub(head, 1); return false; // откат
    }
};

__global__ void q_enq(int *buf, int *h, int *t, int cap, int *succ) { // enqueue kernel
    __shared__ int ok; // shared счетчик
    if (threadIdx.x == 0) ok = 0; __syncthreads(); // 0
    QueueMPMC q; q.init(buf, h, t, cap); // init
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id
    if (q.enqueue(tid)) atomicAdd(&ok, 1); // считаем в shared
    __syncthreads(); if (threadIdx.x == 0) atomicAdd(succ, ok); // 1 atomic на блок
}

__global__ void q_deq(int *buf, int *h, int *t, int cap, int *succ) { // dequeue kernel
    __shared__ int ok; // shared счетчик
    if (threadIdx.x == 0) ok = 0; __syncthreads(); // 0
    QueueMPMC q; q.init(buf, h, t, cap); // init
    int v = -1; // куда брать
    if (q.dequeue(&v)) atomicAdd(&ok, 1); // считаем
    __syncthreads(); if (threadIdx.x == 0) atomicAdd(succ, ok); // 1 atomic на блок
}

double cpu_queue_seq(int n, int cap) { // cpu очередь
    vector<int> q(cap); int h = 0, t = 0; // буфер и указатели
    auto s = chrono::high_resolution_clock::now(); // старт
    for (int i = 0; i < n; i++) if (t < cap) q[t++] = i; // enqueue
    for (int i = 0; i < n; i++) if (h < t) { int x = q[h++]; (void)x; } // dequeue
    auto e = chrono::high_resolution_clock::now(); // конец
    return chrono::duration<double, milli>(e - s).count(); // мс
}

double cpu_stack_seq(int n, int cap) { // cpu стек (для полноты "структур")
    vector<int> st(cap); int top = 0; // стек
    auto s = chrono::high_resolution_clock::now(); // старт
    for (int i = 0; i < n; i++) if (top < cap) st[top++] = i; // push
    for (int i = 0; i < n; i++) if (top > 0) { int x = st[--top]; (void)x; } // pop
    auto e = chrono::high_resolution_clock::now(); // конец
    return chrono::duration<double, milli>(e - s).count(); // мс
}

int main() { // main
    const int threads = 256; // threads
    const int blocks = 2; // blocks
    const int total = threads * blocks; // всего
    const int cap = 256; // емкость

    int *d_buf = nullptr, *d_h = nullptr, *d_t = nullptr, *d_enq = nullptr, *d_deq = nullptr; // указатели
    cuda_check(cudaMalloc(&d_buf, cap * sizeof(int)), "malloc buf"); // malloc
    cuda_check(cudaMalloc(&d_h, sizeof(int)), "malloc head"); // malloc
    cuda_check(cudaMalloc(&d_t, sizeof(int)), "malloc tail"); // malloc
    cuda_check(cudaMalloc(&d_enq, sizeof(int)), "malloc enq"); // malloc
    cuda_check(cudaMalloc(&d_deq, sizeof(int)), "malloc deq"); // malloc
    cuda_check(cudaMemset(d_enq, 0, sizeof(int)), "memset enq"); // 0
    cuda_check(cudaMemset(d_deq, 0, sizeof(int)), "memset deq"); // 0

    cudaEvent_t s1,e1,s2,e2; cudaEventCreate(&s1); cudaEventCreate(&e1); cudaEventCreate(&s2); cudaEventCreate(&e2); // события

    cudaEventRecord(s1); q_enq<<<blocks, threads>>>(d_buf, d_h, d_t, cap, d_enq); cuda_check(cudaGetLastError(), "kernel enq"); cudaEventRecord(e1); cudaEventSynchronize(e1); // enq
    float ms_enq = 0.0f; cudaEventElapsedTime(&ms_enq, s1, e1); // мс

    cudaEventRecord(s2); q_deq<<<blocks, threads>>>(d_buf, d_h, d_t, cap, d_deq); cuda_check(cudaGetLastError(), "kernel deq"); cudaEventRecord(e2); cudaEventSynchronize(e2); // deq
    float ms_deq = 0.0f; cudaEventElapsedTime(&ms_deq, s2, e2); // мс

    int enq = 0, deq = 0; // счетчики
    cuda_check(cudaMemcpy(&enq, d_enq, sizeof(int), cudaMemcpyDeviceToHost), "memcpy enq"); // копия
    cuda_check(cudaMemcpy(&deq, d_deq, sizeof(int), cudaMemcpyDeviceToHost), "memcpy deq"); // копия

    double cpu_q = cpu_queue_seq(total, cap); // cpu очередь
    double cpu_s = cpu_stack_seq(total, cap); // cpu стек

    cout << "mpmc очередь (gpu)\n"; // вывод
    cout << "enqueue успешно: " << enq << "\n"; // вывод
    cout << "dequeue успешно: " << deq << "\n"; // вывод
    cout << "время enqueue+dequeue: " << (ms_enq + ms_deq) << " мс\n\n"; // вывод

    cout << "cpu версии (последовательно)\n"; // вывод
    cout << "cpu очередь время: " << cpu_q << " мс\n"; // вывод
    cout << "cpu стек время: " << cpu_s << " мс\n\n"; // вывод

    cout << "сравнение (cpu/gpu очередь): " << (cpu_q / (ms_enq + ms_deq + 1e-6f)) << "\n"; // вывод

    cudaFree(d_buf); cudaFree(d_h); cudaFree(d_t); cudaFree(d_enq); cudaFree(d_deq); // free
    return 0; // выход
}
