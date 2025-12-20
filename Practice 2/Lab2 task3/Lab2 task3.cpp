#include <iostream> // ввод вывод
#include <random> // рандом
#include <chrono> // время
#include <omp.h> // openmp

using namespace std; // чтобы не писать std::

void copy_array(int* dst, int* src, int n) { // копируем массив
    for (int i = 0; i < n; i++) dst[i] = src[i]; // копия
}

void bubble_sort_seq(int* a, int n) { // пузырек seq
    for (int i = 0; i < n - 1; i++) // внешний цикл
        for (int j = 0; j < n - i - 1; j++) // внутренний цикл
            if (a[j] > a[j + 1]) { // если не по порядку
                int t = a[j]; // временно
                a[j] = a[j + 1]; // обмен
                a[j + 1] = t; // обмен
            }
}

void selection_sort_seq(int* a, int n) { // выборкой seq
    for (int i = 0; i < n - 1; i++) { // внешний цикл
        int min_idx = i; // минимум индекс
        for (int j = i + 1; j < n; j++) // ищем минимум
            if (a[j] < a[min_idx]) min_idx = j; // обновляем минимум
        int t = a[i]; // временно
        a[i] = a[min_idx]; // обмен
        a[min_idx] = t; // обмен
    }
}

void insertion_sort_seq(int* a, int n) { // вставками seq
    for (int i = 1; i < n; i++) { // идем с 1
        int key = a[i]; // текущий
        int j = i - 1; // индекс слева
        while (j >= 0 && a[j] > key) { // сдвигаем
            a[j + 1] = a[j]; // двигаем вправо
            j--; // влево
        }
        a[j + 1] = key; // вставка
    }
}

void bubble_sort_par(int* a, int n) { // пузырек par (odd-even)
    for (int pass = 0; pass < n; pass++) { // проходы
        int start = pass % 2; // 0 или 1
#pragma omp parallel for // параллельно по парам
        for (int j = start; j < n - 1; j += 2) { // шаг 2
            if (a[j] > a[j + 1]) { // если надо менять
                int t = a[j]; // временно
                a[j] = a[j + 1]; // обмен
                a[j + 1] = t; // обмен
            }
        }
    }
}

void selection_sort_par(int* a, int n) { // выборкой par (ищем минимум параллельно)
    for (int i = 0; i < n - 1; i++) { // внешний цикл
        int min_val = a[i]; // минимум значение
        int min_idx = i; // минимум индекс
#pragma omp parallel // параллельная область
        { // блок
            int local_val = min_val; // локальный минимум
            int local_idx = min_idx; // локальный индекс
#pragma omp for nowait // делим цикл между потоками
            for (int j = i + 1; j < n; j++) { // поиск минимума
                if (a[j] < local_val) { // если меньше
                    local_val = a[j]; // запоминаем
                    local_idx = j; // запоминаем
                }
            }
#pragma omp critical // один поток обновляет общий минимум
            { // блок
                if (local_val < min_val) { // если локальный лучше
                    min_val = local_val; // обновляем
                    min_idx = local_idx; // обновляем
                }
            }
        }
        int t = a[i]; // временно
        a[i] = a[min_idx]; // обмен
        a[min_idx] = t; // обмен
    }
}

void insertion_sort_par(int* a, int n) { // вставками par (почти без ускорения)
#pragma omp parallel for // внешний цикл (как в задании)
    for (int i = 1; i < n; i++) { // идем с 1
#pragma omp critical // чтобы не ломать массив
        { // блок
            int key = a[i]; // текущий
            int j = i - 1; // слева
            while (j >= 0 && a[j] > key) { // сдвиг
                a[j + 1] = a[j]; // двигаем
                j--; // влево
            }
            a[j + 1] = key; // вставка
        }
    }
}

double ms_since(chrono::high_resolution_clock::time_point a, chrono::high_resolution_clock::time_point b) { // миллисекунды
    return chrono::duration<double, milli>(b - a).count(); // считаем время
}

int main() { // начало
    int sizes[3] = { 1000, 10000, 100000 }; // размеры тестов
    random_device rd; // источник
    mt19937 gen(rd()); // генератор
    uniform_int_distribution<int> d(1, 100000); // диапазон

    for (int s = 0; s < 3; s++) { // цикл по размерам
        int n = sizes[s]; // текущий размер
        cout << "\nsize = " << n << "\n"; // печать размера
        int* base = new int[n]; // базовый массив
        int* a = new int[n]; // рабочий массив
        for (int i = 0; i < n; i++) base[i] = d(gen); // заполняем базу

        copy_array(a, base, n); // копия
        auto t1 = chrono::high_resolution_clock::now(); // старт
        bubble_sort_seq(a, n); // seq пузырек
        auto t2 = chrono::high_resolution_clock::now(); // конец
        cout << "bubble seq: " << ms_since(t1, t2) << " ms\n"; // вывод

        copy_array(a, base, n); // копия
        auto t3 = chrono::high_resolution_clock::now(); // старт
        bubble_sort_par(a, n); // par пузырек
        auto t4 = chrono::high_resolution_clock::now(); // конец
        cout << "bubble par: " << ms_since(t3, t4) << " ms\n"; // вывод

        copy_array(a, base, n); // копия
        auto t5 = chrono::high_resolution_clock::now(); // старт
        selection_sort_seq(a, n); // seq выборкой
        auto t6 = chrono::high_resolution_clock::now(); // конец
        cout << "select seq: " << ms_since(t5, t6) << " ms\n"; // вывод

        copy_array(a, base, n); // копия
        auto t7 = chrono::high_resolution_clock::now(); // старт
        selection_sort_par(a, n); // par выборкой
        auto t8 = chrono::high_resolution_clock::now(); // конец
        cout << "select par: " << ms_since(t7, t8) << " ms\n"; // вывод

        copy_array(a, base, n); // копия
        auto t9 = chrono::high_resolution_clock::now(); // старт
        insertion_sort_seq(a, n); // seq вставками
        auto t10 = chrono::high_resolution_clock::now(); // конец
        cout << "insert seq: " << ms_since(t9, t10) << " ms\n"; // вывод

        copy_array(a, base, n); // копия
        auto t11 = chrono::high_resolution_clock::now(); // старт
        insertion_sort_par(a, n); // par вставками
        auto t12 = chrono::high_resolution_clock::now(); // конец
        cout << "insert par: " << ms_since(t11, t12) << " ms\n"; // вывод

        delete[] base; // чистим
        delete[] a; // чистим
        base = nullptr; // на всякий
        a = nullptr; // на всякий
    }
    return 0; // конец
}
