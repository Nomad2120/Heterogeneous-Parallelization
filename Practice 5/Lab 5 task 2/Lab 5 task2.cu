%%writefile task2.cu
#include <iostream> // вывод
#include <vector> // векторы для проверки
#include <cuda_runtime.h> // cuda

using namespace std; // чтоб не писать std::

void cuda_check(cudaError_t err, const char* place) { // проверка cuda
    if (err != cudaSuccess) { // если ошибка
        cout << "ошибка cuda в " << place << ": " << cudaGetErrorString(err) << "\n"; // вывод
        exit(1); // выходим
    }
}

struct Stack { // стек
    int *data; // массив
    int *top; // размер стека
    int capacity; // емкость

    __device__ void init(int *buffer, int *top_ptr, int size) { // init
        data = buffer; // буфер
        top = top_ptr; // top
        capacity = size; // емкость
        if (blockIdx.x == 0 && threadIdx.x == 0) *top = 0; // пусто
        __syncthreads(); // барьер
    }

    __device__ bool push(int value) { // push
        int pos = atomicAdd(top, 1); // позиция
        if (pos < capacity) { // если влезли
            data[pos] = value; // кладем
            return true; // успех
        }
        atomicSub(top, 1); // откат
        return false; // неуспех
    }

    __device__ bool pop(int *value) { // pop
        int pos = atomicSub(top, 1) - 1; // индекс
        if (pos >= 0) { // если не пусто
            *value = data[pos]; // забрали
            return true; // успех
        }
        atomicAdd(top, 1); // откат
        return false; // неуспех
    }
};

struct Queue { // очередь
    int *data; // массив
    int *head; // голова
    int *tail; // хвост
    int capacity; // емкость

    __device__ void init(int *buffer, int *head_ptr, int *tail_ptr, int size) { // init
        data = buffer; // буфер
        head = head_ptr; // head
        tail = tail_ptr; // tail
        capacity = size; // емкость
        if (blockIdx.x == 0 && threadIdx.x == 0) { // один поток
            *head = 0; // head=0
            *tail = 0; // tail=0
        }
        __syncthreads(); // барьер
    }

    __device__ bool enqueue(int value) { // enqueue
        int pos = atomicAdd(tail, 1); // место в хвосте
        if (pos < capacity) { // если влезли
            data[pos] = value; // кладем
            return true; // успех
        }
        atomicSub(tail, 1); // откат если переполнение
        return false; // неуспех
    }

    __device__ bool dequeue(int *value) { // dequeue
        int pos = atomicAdd(head, 1); // берем позицию в голове
        int tail_now = atomicAdd(tail, 0); // читаем tail атомарно
        if (pos < tail_now) { // если есть элементы
            *value = data[pos]; // забираем
            return true; // успех
        }
        atomicSub(head, 1); // откат если пусто
        return false; // неуспех
    }
};

__global__ void stack_kernel(int *buf, int *top_ptr, int capacity, int *push_ok, int *pop_ok, int *pop_val) { // ядро стека
    Stack st; // стек
    st.init(buf, top_ptr, capacity); // init

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id
    int v = -1; // значение pop

    if ((tid % 2) == 0) { // четные push
        bool ok = st.push(tid); // push
        push_ok[tid] = ok ? 1 : 0; // записали
        pop_ok[tid] = 0; // pop не делали
        pop_val[tid] = -1; // нет
    } else { // нечетные pop
        bool ok = st.pop(&v); // pop
        push_ok[tid] = 0; // push не делали
        pop_ok[tid] = ok ? 1 : 0; // записали
        pop_val[tid] = ok ? v : -1; // значение
    }
}

__global__ void queue_kernel(int *buf, int *head_ptr, int *tail_ptr, int capacity, int *enq_ok, int *deq_ok, int *deq_val) { // ядро очереди
    Queue q; // очередь
    q.init(buf, head_ptr, tail_ptr, capacity); // init

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // id
    int v = -1; // значение dequeue

    if ((tid % 2) == 0) { // четные enqueue
        bool ok = q.enqueue(tid); // enqueue
        enq_ok[tid] = ok ? 1 : 0; // записали
        deq_ok[tid] = 0; // deq не делали
        deq_val[tid] = -1; // нет
    } else { // нечетные dequeue
        bool ok = q.dequeue(&v); // dequeue
        enq_ok[tid] = 0; // enq не делали
        deq_ok[tid] = ok ? 1 : 0; // записали
        deq_val[tid] = ok ? v : -1; // значение
    }
}

int main() { // main
    const int threads = 256; // потоков в блоке
    const int blocks = 2; // блоков
    const int total = threads * blocks; // всего потоков
    const int capacity = 256; // емкость (одинаково для стека и очереди)

    int *d_sbuf = nullptr; // буфер стека
    int *d_top = nullptr; // top
    int *d_push_ok = nullptr; // push ok
    int *d_pop_ok = nullptr; // pop ok
    int *d_pop_val = nullptr; // pop val

    int *d_qbuf = nullptr; // буфер очереди
    int *d_head = nullptr; // head
    int *d_tail = nullptr; // tail
    int *d_enq_ok = nullptr; // enq ok
    int *d_deq_ok = nullptr; // deq ok
    int *d_deq_val = nullptr; // deq val

    cuda_check(cudaMalloc(&d_sbuf, capacity * sizeof(int)), "malloc stack buf"); // malloc
    cuda_check(cudaMalloc(&d_top, sizeof(int)), "malloc top"); // malloc
    cuda_check(cudaMalloc(&d_push_ok, total * sizeof(int)), "malloc push_ok"); // malloc
    cuda_check(cudaMalloc(&d_pop_ok, total * sizeof(int)), "malloc pop_ok"); // malloc
    cuda_check(cudaMalloc(&d_pop_val, total * sizeof(int)), "malloc pop_val"); // malloc

    cuda_check(cudaMalloc(&d_qbuf, capacity * sizeof(int)), "malloc queue buf"); // malloc
    cuda_check(cudaMalloc(&d_head, sizeof(int)), "malloc head"); // malloc
    cuda_check(cudaMalloc(&d_tail, sizeof(int)), "malloc tail"); // malloc
    cuda_check(cudaMalloc(&d_enq_ok, total * sizeof(int)), "malloc enq_ok"); // malloc
    cuda_check(cudaMalloc(&d_deq_ok, total * sizeof(int)), "malloc deq_ok"); // malloc
    cuda_check(cudaMalloc(&d_deq_val, total * sizeof(int)), "malloc deq_val"); // malloc

    cudaEvent_t e1s, e1e, e2s, e2e; // события для времени
    cuda_check(cudaEventCreate(&e1s), "event create"); // create
    cuda_check(cudaEventCreate(&e1e), "event create"); // create
    cuda_check(cudaEventCreate(&e2s), "event create"); // create
    cuda_check(cudaEventCreate(&e2e), "event create"); // create

    cuda_check(cudaEventRecord(e1s), "event record"); // старт стека
    stack_kernel<<<blocks, threads>>>(d_sbuf, d_top, capacity, d_push_ok, d_pop_ok, d_pop_val); // запуск стека
    cuda_check(cudaGetLastError(), "kernel stack launch"); // проверка
    cuda_check(cudaEventRecord(e1e), "event record"); // конец стека
    cuda_check(cudaEventSynchronize(e1e), "event sync"); // ждем
    float ms_stack = 0.0f; // время стека
    cuda_check(cudaEventElapsedTime(&ms_stack, e1s, e1e), "elapsed"); // считаем

    cuda_check(cudaEventRecord(e2s), "event record"); // старт очереди
    queue_kernel<<<blocks, threads>>>(d_qbuf, d_head, d_tail, capacity, d_enq_ok, d_deq_ok, d_deq_val); // запуск очереди
    cuda_check(cudaGetLastError(), "kernel queue launch"); // проверка
    cuda_check(cudaEventRecord(e2e), "event record"); // конец очереди
    cuda_check(cudaEventSynchronize(e2e), "event sync"); // ждем
    float ms_queue = 0.0f; // время очереди
    cuda_check(cudaEventElapsedTime(&ms_queue, e2s, e2e), "elapsed"); // считаем

    vector<int> h_push_ok(total), h_pop_ok(total), h_pop_val(total); // cpu стек
    vector<int> h_enq_ok(total), h_deq_ok(total), h_deq_val(total); // cpu очередь
    int h_top = 0; // top cpu
    int h_head = 0; // head cpu
    int h_tail = 0; // tail cpu

    cuda_check(cudaMemcpy(h_push_ok.data(), d_push_ok, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy push_ok"); // копия
    cuda_check(cudaMemcpy(h_pop_ok.data(), d_pop_ok, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy pop_ok"); // копия
    cuda_check(cudaMemcpy(h_pop_val.data(), d_pop_val, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy pop_val"); // копия
    cuda_check(cudaMemcpy(&h_top, d_top, sizeof(int), cudaMemcpyDeviceToHost), "memcpy top"); // копия

    cuda_check(cudaMemcpy(h_enq_ok.data(), d_enq_ok, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy enq_ok"); // копия
    cuda_check(cudaMemcpy(h_deq_ok.data(), d_deq_ok, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy deq_ok"); // копия
    cuda_check(cudaMemcpy(h_deq_val.data(), d_deq_val, total * sizeof(int), cudaMemcpyDeviceToHost), "memcpy deq_val"); // копия
    cuda_check(cudaMemcpy(&h_head, d_head, sizeof(int), cudaMemcpyDeviceToHost), "memcpy head"); // копия
    cuda_check(cudaMemcpy(&h_tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost), "memcpy tail"); // копия

    int push_uspeh = 0; // stack push успешно
    int pop_uspeh = 0; // stack pop успешно
    int pop_neuspeh = 0; // stack pop неуспешно
    int pop_oshibka = 0; // stack pop ошибка

    vector<int> spushed(total, 0); // что клали в стек
    vector<int> spopped(total, 0); // что вынули из стека

    for (int i = 0; i < total; i++) { // считаем push
        if (h_push_ok[i] == 1) { // если push
            spushed[i] = 1; // отметка
            push_uspeh++; // плюс
        }
    }

    for (int i = 0; i < total; i++) { // проверяем pop
        if ((i % 2) == 1) { // только нечетные делали pop
            if (h_pop_ok[i] == 1) { // pop успешен
                int val = h_pop_val[i]; // значение
                pop_uspeh++; // плюс
                if (val < 0 || val >= total) { pop_oshibka++; continue; } // мусор
                if (spushed[val] == 0) { pop_oshibka++; continue; } // не клали
                if (spopped[val] == 1) { pop_oshibka++; continue; } // дубль
                spopped[val] = 1; // отметка
            } else { // pop неуспешен
                pop_neuspeh++; // плюс
            }
        }
    }

    int expected_top = push_uspeh - pop_uspeh; // ожидание top
    if (expected_top < 0) expected_top = 0; // защита
    bool stack_ok = (pop_oshibka == 0 && h_top == expected_top); // проверка стека

    int enq_uspeh = 0; // queue enqueue успешно
    int deq_uspeh = 0; // queue dequeue успешно
    int deq_neuspeh = 0; // queue dequeue неуспешно
    int deq_oshibka = 0; // queue dequeue ошибка

    vector<int> qpushed(total, 0); // что клали в очередь
    vector<int> qpopped(total, 0); // что вынули из очереди

    for (int i = 0; i < total; i++) { // считаем enqueue
        if (h_enq_ok[i] == 1) { // если enqueue
            qpushed[i] = 1; // отметка
            enq_uspeh++; // плюс
        }
    }

    for (int i = 0; i < total; i++) { // проверяем dequeue
        if ((i % 2) == 1) { // только нечетные делали dequeue
            if (h_deq_ok[i] == 1) { // dequeue успешен
                int val = h_deq_val[i]; // значение
                deq_uspeh++; // плюс
                if (val < 0 || val >= total) { deq_oshibka++; continue; } // мусор
                if (qpushed[val] == 0) { deq_oshibka++; continue; } // не клали
                if (qpopped[val] == 1) { deq_oshibka++; continue; } // дубль
                qpopped[val] = 1; // отметка
            } else { // dequeue неуспешен
                deq_neuspeh++; // плюс
            }
        }
    }

    int expected_tail = enq_uspeh; // в простой модели tail == сколько добавили
    bool queue_ok = (deq_oshibka == 0 && h_tail == expected_tail && h_head == deq_uspeh); // проверка очереди

    cout << "потоков: " << total << "\n"; // вывод
    cout << "емкость: " << capacity << "\n"; // вывод

    cout << "стек время: " << ms_stack << " мс\n"; // время стека
    cout << "стек push успешно: " << push_uspeh << "\n"; // push
    cout << "стек pop успешно:  " << pop_uspeh << "\n"; // pop
    cout << "стек pop неуспешно: " << pop_neuspeh << "\n"; // pop fail
    cout << "стек pop ошибка:    " << pop_oshibka << "\n"; // ошибка
    cout << "стек top: " << h_top << "\n"; // top
    cout << "стек проверка: " << (stack_ok ? "успешно" : "неуспешно") << "\n\n"; // итог

    cout << "очередь время: " << ms_queue << " мс\n"; // время очереди
    cout << "очередь enqueue успешно: " << enq_uspeh << "\n"; // enq
    cout << "очередь dequeue успешно:  " << deq_uspeh << "\n"; // deq
    cout << "очередь dequeue неуспешно: " << deq_neuspeh << "\n"; // deq fail
    cout << "очередь dequeue ошибка:    " << deq_oshibka << "\n"; // ошибка
    cout << "очередь head: " << h_head << "\n"; // head
    cout << "очередь tail: " << h_tail << "\n"; // tail
    cout << "очередь проверка: " << (queue_ok ? "успешно" : "неуспешно") << "\n\n"; // итог

    if (ms_queue > 0.0f) { // защита
        cout << "сравнение: стек/очередь = " << (ms_stack / ms_queue) << "\n"; // отношение
    } else { // если вдруг 0
        cout << "сравнение: стек/очередь = 0\n"; // вывод
    }

    cuda_check(cudaEventDestroy(e1s), "event destroy"); // destroy
    cuda_check(cudaEventDestroy(e1e), "event destroy"); // destroy
    cuda_check(cudaEventDestroy(e2s), "event destroy"); // destroy
    cuda_check(cudaEventDestroy(e2e), "event destroy"); // destroy

    cuda_check(cudaFree(d_sbuf), "free sbuf"); // free
    cuda_check(cudaFree(d_top), "free top"); // free
    cuda_check(cudaFree(d_push_ok), "free push_ok"); // free
    cuda_check(cudaFree(d_pop_ok), "free pop_ok"); // free
    cuda_check(cudaFree(d_pop_val), "free pop_val"); // free

    cuda_check(cudaFree(d_qbuf), "free qbuf"); // free
    cuda_check(cudaFree(d_head), "free head"); // free
    cuda_check(cudaFree(d_tail), "free tail"); // free
    cuda_check(cudaFree(d_enq_ok), "free enq_ok"); // free
    cuda_check(cudaFree(d_deq_ok), "free deq_ok"); // free
    cuda_check(cudaFree(d_deq_val), "free deq_val"); // free

    return 0; // выход
}
