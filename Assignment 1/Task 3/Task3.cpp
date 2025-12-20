#include <iostream> // ввод вывод
#include <random> // рандом
#include <chrono> // время
#include <omp.h> // openmp
using namespace std; // чтобы не писать std::

int main() { // начало программы
    int n = 1000000; // размер массива
    int* arr = new int[n]; // динамически выделяем массив
    random_device rd; // источник
    mt19937 gen(rd()); // генератор
    uniform_int_distribution<int> d(1, 100); // диапазон 1..100
    for (int i = 0; i < n; i++) arr[i] = d(gen); // заполняем массив
    auto s1 = chrono::high_resolution_clock::now(); // старт seq
    int min_seq = arr[0]; // минимум seq
    int max_seq = arr[0]; // максимум seq

    for (int i = 1; i < n; i++) { // последовательный проход
        if (arr[i] < min_seq) min_seq = arr[i]; // обновляем минимум
        if (arr[i] > max_seq) max_seq = arr[i]; // обновляем максимум
    }

    auto e1 = chrono::high_resolution_clock::now(); // конец seq
    double t_seq = chrono::duration<double, milli>(e1 - s1).count(); // время seq
    auto s2 = chrono::high_resolution_clock::now(); // старт par
    int min_par = 101; // минимум par (так как числа 1..100)
    int max_par = 0; // максимум par

#pragma omp parallel // параллельная область
    { // блок
        int local_min = 101; // локальный минимум
        int local_max = 0; // локальный максимум
#pragma omp for nowait // делим цикл между потоками
        for (int i = 0; i < n; i++) { // проход по массиву
            if (arr[i] < local_min) local_min = arr[i]; // локальный минимум
            if (arr[i] > local_max) local_max = arr[i]; // локальный максимум
        }
#pragma omp critical // обновляем общий min/max
        { // блок
            if (local_min < min_par) min_par = local_min; // общий минимум
            if (local_max > max_par) max_par = local_max; // общий максимум
        }
    }
    auto e2 = chrono::high_resolution_clock::now(); // конец par
    double t_par = chrono::duration<double, milli>(e2 - s2).count(); // время par

    cout << "seq min = " << min_seq << ", seq max = " << max_seq << "\n"; // вывод seq
    cout << "par min = " << min_par << ", par max = " << max_par << "\n"; // вывод par
    cout << "seq time = " << t_seq << " ms\n"; // время seq
    cout << "par time = " << t_par << " ms\n"; // время par

    delete[] arr; // освобождаем память
    arr = nullptr; // обнуляем указатель
    return 0; // конец программы
}
