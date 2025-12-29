#include <iostream> // ввод вывод
#include <cstdlib> // rand srand
#include <ctime> // time
#include <chrono> // время
#include <omp.h> // openmp

using namespace std;  // чтоб не писать std::

void fill_array(int* arr, int n) { // заполняем массив
    for (int i = 0; i < n; i++) { // цикл
        arr[i] = rand() % 10000 + 1; // случайные числа
    }
}

void copy_array(int* to, int* from, int n) { // копия массива
    for (int i = 0; i < n; i++) { // цикл
        to[i] = from[i]; // копируем
    }
}

void selection_sort_seq(int* arr, int n) { // выбором seq
    for (int i = 0; i < n - 1; i++) { // идем по массиву
        int min_idx = i; // считаем что это минимум

        for (int j = i + 1; j < n; j++) { // ищем меньше
            if (arr[j] < arr[min_idx]) { // если нашли
                min_idx = j; // запоминаем
            }
        }

        int temp = arr[i]; // временно сохраняем
        arr[i] = arr[min_idx]; // ставим минимум
        arr[min_idx] = temp; // возвращаем обратно
    }
}

void selection_sort_par(int* arr, int n) { // выбором par
    for (int i = 0; i < n - 1; i++) { // внешний цикл
        int min_val = arr[i]; // минимум
        int min_idx = i; // индекс минимума

#pragma omp parallel
        { // параллельная зона
            int local_min = min_val; // локальный минимум
            int local_idx = min_idx; // локальный индекс

#pragma omp for nowait
            for (int j = i + 1; j < n; j++) { // делим массив
                if (arr[j] < local_min) { // если меньше
                    local_min = arr[j]; // обновляем
                    local_idx = j; // обновляем
                }
            }

#pragma omp critical
            { // обновляем общий минимум
                if (local_min < min_val) {
                    min_val = local_min;
                    min_idx = local_idx;
                }
            }
        }

        int temp = arr[i]; // временно сохраняем
        arr[i] = arr[min_idx]; // ставим минимум
        arr[min_idx] = temp; // возвращаем обратно
    }
}

void test_size(int n) { // тест размера
    int* base = new int[n]; // исходный массив
    int* arr = new int[n]; // рабочий массив

    fill_array(base, n); // заполняем

    copy_array(arr, base, n); // копия
    auto s1 = chrono::high_resolution_clock::now(); // старт seq
    selection_sort_seq(arr, n); // сортировка
    auto e1 = chrono::high_resolution_clock::now(); // конец seq
    double t_seq = chrono::duration<double, milli>(e1 - s1).count(); // время

    copy_array(arr, base, n); // копия
    auto s2 = chrono::high_resolution_clock::now(); // старт par
    selection_sort_par(arr, n); // сортировка
    auto e2 = chrono::high_resolution_clock::now(); // конец par
    double t_par = chrono::duration<double, milli>(e2 - s2).count(); // время

    cout << "\nразмер = " << n << "\n"; // вывод
    cout << "последовательно: " << t_seq << " мс\n"; // seq
    cout << "параллельно:     " << t_par << " мс\n"; // par

    delete[] base; // чистим
    delete[] arr; // чистим
}

int main() { // main
    srand(time(0)); // рандом
    test_size(1000); // тест 1000
    test_size(10000); // тест 10000
    return 0; // выход
}
