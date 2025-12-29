#include <iostream> // ввод и вывод
#include <cstdlib>  // rand, srand
#include <ctime> // time
#include <chrono> // измерение времени
#include <omp.h>  // openmp

using namespace std;  // чтоб не писать std::

void task2() { // функция
    const int N = 10000; // размер массива
    int* arr = new int[N]; // динамический массив
    srand(time(0)); // инициализация рандома

    for (int i = 0; i < N; i++) { // заполняем массив
        arr[i] = rand() % 10000 + 1; // числа от 1 до 10000
    }
    
  // последовательный поиск
    auto s1 = chrono::high_resolution_clock::now(); // старт seq
    int min_seq = arr[0]; // минимум
    int max_seq = arr[0]; // максимум

    for (int i = 1; i < N; i++) { // проход по массиву
        if (arr[i] < min_seq) min_seq = arr[i]; // минимум
        if (arr[i] > max_seq) max_seq = arr[i]; // максимум
    }
 // параллельный поиск
    auto e1 = chrono::high_resolution_clock::now(); // конец seq
    double time_seq = chrono::duration<double, milli>(e1 - s1).count(); // время

    auto s2 = chrono::high_resolution_clock::now(); // старт par
    int min_par = 10001; // минимум
    int max_par = 0; // максимум

#pragma omp parallel
    { // параллельная область
        int local_min = 10001; // локальный минимум
        int local_max = 0; // локальный максимум

#pragma omp for nowait
        for (int i = 0; i < N; i++) { // делим массив
            if (arr[i] < local_min) local_min = arr[i]; // локальный минимум
            if (arr[i] > local_max) local_max = arr[i]; // локальный максимум
        }

#pragma omp critical
        { // обновляем результат
            if (local_min < min_par) min_par = local_min; // общий минимум
            if (local_max > max_par) max_par = local_max; // общий максимум
        }
    }

    auto e2 = chrono::high_resolution_clock::now(); // конец par
    double time_par = chrono::duration<double, milli>(e2 - s2).count(); // время
// выводы
    cout << "последовательно:\n"; // вывод
    cout << "минимум = " << min_seq << "\n"; // минимум
    cout << "максимум = " << max_seq << "\n"; // максимум
    cout << "время = " << time_seq << " мс\n\n"; // время

    cout << "параллельно:\n"; // вывод
    cout << "минимум = " << min_par << "\n"; // минимум
    cout << "максимум = " << max_par << "\n"; // максимум
    cout << "время = " << time_par << " мс\n"; // время

    delete[] arr; // освобождаем память
    arr = nullptr; // обнуляем указатель
}

int main() { // main
    task2(); // запуск задания
    return 0; // выход
}
