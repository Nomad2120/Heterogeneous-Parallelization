#include <iostream> // ввод вывод
#include <random>   // рандом
#include <chrono>   // время
using namespace std; // чтобы не писать std::

int main() { // начало программы
    int n = 1000000; // размер массива
    int* arr = new int[n]; // динамически выделяем массив
    random_device rd; // источник
    mt19937 gen(rd()); // генератор
    uniform_int_distribution<int> d(1, 100); // диапазон 1..100
    for (int i = 0; i < n; i++) arr[i] = d(gen); // заполняем массив
    auto start = chrono::high_resolution_clock::now(); // начало замера

    int min_val = arr[0]; // начальный минимум
    int max_val = arr[0]; // начальный максимум
    for (int i = 1; i < n; i++) { // последовательный проход
        if (arr[i] < min_val) min_val = arr[i]; // обновляем минимум
        if (arr[i] > max_val) max_val = arr[i]; // обновляем максимум
    }
    auto end = chrono::high_resolution_clock::now(); // конец замера
    double time_ms = chrono::duration<double, milli>(end - start).count(); // время в мс
    cout << "min value = " << min_val << "\n"; // вывод минимума
    cout << "max value = " << max_val << "\n"; // вывод максимума
    cout << "time = " << time_ms << " ms\n";   // вывод времени

    delete[] arr; // освобождаем память
    arr = nullptr; // обнуляем указатель
    return 0; // конец программы
}
