#include <iostream> // ввод вывод
#include <random> // рандом
#include <omp.h>  // openmp

using namespace std; // чтобы не писать std::

void fill_random(int* arr, int n) { // заполняем массив
    random_device rd; // источник
    mt19937 gen(rd()); // генератор
    uniform_int_distribution<int> d(1, 100); // диапазон
    for (int i = 0; i < n; i++) arr[i] = d(gen); // заполнение
}

void print_first(int* arr, int n) { // печать первых 10
    int k = (n < 10 ? n : 10); // сколько печатать
    for (int i = 0; i < k; i++) cout << arr[i] << " "; // вывод
    cout << "\n"; // перенос
}

void print_last(int* arr, int n) { // печать последних 10
    int start = (n > 10 ? n - 10 : 0); // откуда печатать
    for (int i = start; i < n; i++) cout << arr[i] << " "; // вывод
    cout << "\n"; // перенос
}

void bubble_sort_parallel(int* arr, int n) { // пузырек odd-even
    for (int pass = 0; pass < n; pass++) { // проходы
        int start = pass % 2; // чет или нечет
#pragma omp parallel for // параллельный цикл
        for (int j = start; j < n - 1; j += 2) { // по парам
            if (arr[j] > arr[j + 1]) { // если не по порядку
                int t = arr[j]; // обмен
                arr[j] = arr[j + 1]; // обмен
                arr[j + 1] = t; // обмен
            }
        }
    }
}

int main() { // начало программы
    int n; // размер
    cout << "array size: "; // сообщение
    cin >> n; // ввод
    if (n <= 0) return 1; // проверка
    int* arr = new int[n]; // динамический массив
    fill_random(arr, n); // заполняем
    cout << "first before: "; // до сортировки
    print_first(arr, n); // печать
    cout << "last before:  "; // до сортировки
    print_last(arr, n); // печать
    bubble_sort_parallel(arr, n); // сортировка
    cout << "first after:  "; // после сортировки
    print_first(arr, n); // печать
    cout << "last after:   "; // после сортировки
    print_last(arr, n); // печать
    delete[] arr; // освобождаем память
    arr = nullptr; // обнуляем
    return 0; // конец
}
