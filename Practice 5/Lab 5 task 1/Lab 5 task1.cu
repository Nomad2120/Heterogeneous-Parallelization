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

struct Stack { // стек
    int *data; // данные
    int *top; // top это размер стека
    int capacity; // емкость

    __device__ void init(int *buffer, int *top_ptr, int size) { // init
        data = buffer; // буфер
        top = top_ptr; // top
        capacity = size; // емкость
        if (blockIdx.x == 0 && threadIdx.x == 0) *top = 0; // пусто
        __syncthreads(); // барьер
    }

    __device__ bool push(int value) { // push
        int pos = atomicAdd(top, 1); // берем позицию
        if (pos < capacity) { // если влезли
            data[pos] = value; // кладем
            return true; // успех
        }
        atomicSub(top, 1); // откат
        return false; // неуспех
    }

    __device__ bool pop(int *value) { // pop
        int pos = atomicSub(top, 1) - 1; // берем индекс
        if (pos >= 0) { // если не пусто
            *value = data[pos]; // забираем
            return true; // успех
        }
        atomicAdd(top, 1); // откат
        return false; // неуспех
    }
};

__global__ void stack_kernel(int *stack_buf, int *top_ptr, int capacity, int *push_ok, int *pop_ok, int *pop_val) { // ядро
    Stack st; // стек
    st.init(stack_buf, top_ptr, capacity); // init

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id
    int v = -1; // pop value

    if ((tid % 2) == 0) { // четные push
        bool ok = st.push(tid); // push
        push_ok[tid] = ok ? 1 : 0; // успех push
        pop_ok[tid] = 0; // pop не делали
        pop_val[tid] = -1; // нет значения
    } else { // нечетные pop
        bool ok = st.pop(&v); // pop
        push_ok[tid] = 0; // push не делали
        pop_ok[tid] = ok ? 1 : 0; // успех pop
        pop_val[tid] = ok ? v : -1; // значение
    }
}

int main() { // main
    const int threads = 256; // потоков
    const int blocks = 2; // блоков
    const int total = threads * blocks; // всего
    const int capacity = 256; // емкость

    int *d_stack = nullptr; // стек gpu
    int *d_top = nullptr; // top gpu
    int *d_push_ok = nullptr; // push ok
    int *d_pop_ok = nullptr; // pop ok
    int *d_pop_val = nullptr; // pop val

    cuda_check(cudaMalloc(&d_stack, capacity * sizeof(int)), "cudaMalloc stack"); // malloc
    cuda_check(cudaMalloc(&d_top, sizeof(int)), "cudaMalloc top"); // malloc
    cuda_check(cudaMalloc(&d_push_ok, total * sizeof(int)), "cudaMalloc push_ok"); // malloc
    cuda_check(cudaMalloc(&d_pop_ok, total * sizeof(int)), "cudaMalloc pop_ok"); // malloc
    cuda_check(cudaMalloc(&d_pop_val, total * sizeof(int)), "cudaMalloc pop_val"); // malloc

    stack_kernel<<<blocks, threads>>>(d_stack, d_top, capacity, d_push_ok, d_pop_ok, d_pop_val); // запуск
    cuda_check(cudaGetLastError(), "kernel launch"); // проверка
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ждем

    vector<int> h_push_ok(total); // cpu push ok
    vector<int> h_pop_ok(total); // cpu pop ok
    vector<int> h_pop_val(total); // cpu pop val
    int h_top = 0; // cpu top

    cuda_check(cudaMemcpy(h_push_ok.data(), d_push_ok, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy push_ok"); // копия
    cuda_check(cudaMemcpy(h_pop_ok.data(), d_pop_ok, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy pop_ok"); // копия
    cuda_check(cudaMemcpy(h_pop_val.data(), d_pop_val, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy pop_val"); // копия
    cuda_check(cudaMemcpy(&h_top, d_top, sizeof(int), cudaMemcpyDeviceToHost), "memcpy top"); // копия

    vector<int> pushed(total, 0); // какие значения были pushed
    vector<int> popped(total, 0); // какие значения уже popped
    int push_uspeh = 0; // push успешно
    int pop_uspeh = 0; // pop успешно
    int pop_neuspeh = 0; // pop неуспешно
    int pop_oshibka = 0; // pop ошибка (мусор или дубль)

    for (int i = 0; i < total; i++) { // считаем push
        if (h_push_ok[i] == 1) { // push ok
            pushed[i] = 1; // помечаем
            push_uspeh++; // плюс
        }
    }

    for (int i = 0; i < total; i++) { // проверяем pop
        if (h_pop_ok[i] == 1) { // pop ok
            int val = h_pop_val[i]; // значение
            pop_uspeh++; // плюс
            if (val < 0 || val >= total) { pop_oshibka++; continue; } // мусор
            if (pushed[val] == 0) { pop_oshibka++; continue; } // не клали
            if (popped[val] == 1) { pop_oshibka++; continue; } // дубль
            popped[val] = 1; // помечаем
        } else { // pop fail
            if ((i % 2) == 1) pop_neuspeh++; // только нечетные реально делали pop
        }
    }

    int expected_top = push_uspeh - pop_uspeh; // ожидаемо
    if (expected_top < 0) expected_top = 0; // защита

    cout << "потоков: " << total << "\n"; // вывод
    cout << "емкость: " << capacity << "\n"; // вывод
    cout << "push успешно: " << push_uspeh << "\n"; // вывод
    cout << "pop успешно:  " << pop_uspeh << "\n"; // вывод
    cout << "pop неуспешно: " << pop_neuspeh << "\n"; // вывод
    cout << "pop ошибка:    " << pop_oshibka << "\n"; // вывод
    cout << "top после ядра: " << h_top << "\n"; // вывод
    cout << "ожидалось top:  " << expected_top << "\n"; // вывод

    if (pop_oshibka == 0 && h_top == expected_top) { // если все ок
        cout << "проверка: успешно\n"; // итог
    } else { // иначе
        cout << "проверка: неуспешно\n"; // итог
    }

    cuda_check(cudaFree(d_stack), "cudaFree stack"); // free
    cuda_check(cudaFree(d_top), "cudaFree top"); // free
    cuda_check(cudaFree(d_push_ok), "cudaFree push_ok"); // free
    cuda_check(cudaFree(d_pop_ok), "cudaFree pop_ok"); // free
    cuda_check(cudaFree(d_pop_val), "cudaFree pop_val"); // free

    return 0; // выход
}
