#include <iostream> // вывод
#include <cstdlib> // rand srand
#include <ctime> // время
#include <chrono> // время
#include <cuda_runtime.h> // cuda

using namespace std; // чтоб не писать std::

__global__ void sort_block(int* data, int n, int block_size) { // сортим кусок на gpu
    int bid = blockIdx.x; // номер блока
    int start = bid * block_size; // начало куска
    int end = start + block_size; // конец куска
    if (end > n) end = n; // чтобы не вылезти за массив

    for (int i = start; i < end; i++) { // простой пузырек
        for (int j = start; j + 1 < end; j++) { // бегаем по соседям
            if (data[j] > data[j + 1]) { // если перепутаны
                int temp = data[j]; // временно
                data[j] = data[j + 1]; // меняем
                data[j + 1] = temp; // обратно
            }
        }
    }
}

void merge_cpu(int* arr, int n, int block_size) { // сливаем куски на cpu
    for (int size = block_size; size < n; size *= 2) { // размер кусков растет
        for (int left = 0; left < n; left += 2 * size) { // берем 2 куска
            int mid = left + size; // середина
            int right = left + 2 * size; // конец
            if (mid > n) mid = n; // подрезаем границу
            if (right > n) right = n; // тоже подрезаем
            if (mid >= right) continue; // если второго куска нет

            int len = right - left; // длина участка
            int* temp = new int[len]; // временный массив
            int i = left; // левый кусок
            int j = mid; // правый кусок
            int k = 0; // позиция в temp

            while (i < mid && j < right) { // пока есть с двух сторон
                if (arr[i] < arr[j]) temp[k++] = arr[i++]; // берем слева
                else temp[k++] = arr[j++]; // берем справа
            }
            while (i < mid) temp[k++] = arr[i++]; // дописываем левый хвост
            while (j < right) temp[k++] = arr[j++]; // дописываем правый хвост

            for (int x = 0; x < len; x++) arr[left + x] = temp[x]; // копируем назад
            delete[] temp; // освобождаем память
        }
    }
}

void test_size(int n) { // тест размера
    int* arr = new int[n]; // массив на cpu
    for (int i = 0; i < n; i++) arr[i] = rand() % 10000 + 1; // заполняем числами

    int* d_arr = nullptr; // массив на gpu
    cudaMalloc(&d_arr, n * (int)sizeof(int)); // память на gpu
    cudaMemcpy(d_arr, arr, n * (int)sizeof(int), cudaMemcpyHostToDevice); // копируем на gpu

    int block_size = 256; // сколько элементов в куске
    int blocks = (n + block_size - 1) / block_size; // сколько блоков нужно

    auto s = chrono::high_resolution_clock::now(); // старт времени

    sort_block<<<blocks, 1>>>(d_arr, n, block_size); // каждый блок сортит свой кусок
    cudaDeviceSynchronize(); // ждем пока gpu закончит

    cudaMemcpy(arr, d_arr, n * (int)sizeof(int), cudaMemcpyDeviceToHost); // обратно на cpu
    merge_cpu(arr, n, block_size); // сливаем все в один массив

    auto e = chrono::high_resolution_clock::now(); // конец времени
    double t = chrono::duration<double, milli>(e - s).count(); // время в мс

    cout << "размер = " << n << ", время = " << t << " мс\n"; // вывод

    cudaFree(d_arr); // чистим gpu память
    delete[] arr; // чистим cpu память
    arr = nullptr; // на всякий
}

int main() { // main
    srand(time(0)); // рандом
    test_size(10000); // тест 10 000
    test_size(100000); // тест 100 000
    return 0; // выход
}

