#include <iostream> // ввод вывод
#include <random> // рандом
#include <chrono> // время
#include <omp.h>  // openmp
using namespace std; // чтобы не писать std::

double average_sequential(int* arr, int n) { // обычное среднее
    long long sum = 0;                       // сумма
    for (int i = 0; i < n; i++) sum += arr[i]; // складываем
    return (double)sum / n;                  // среднее
}

double average_parallel(int* arr, int n) {   // среднее с openmp
    long long sum = 0;                       // общая сумма
#pragma omp parallel for reduction(+:sum)    // параллельный цикл
    for (int i = 0; i < n; i++) sum += arr[i]; // каждый поток считает
    return (double)sum / n;                  // среднее
}

int main() {                                 // начало программы
    int n;                                   // размер массива
    cout << "array size: ";                  // вывод текста
    cin >> n;                                // ввод размера
    if (n <= 0) return 1;                    // простая проверка
    int* arr = new int[n];                   // динамический массив
    random_device rd;                        // источник
    mt19937 gen(rd());                       // генератор
    uniform_int_distribution<int> d(1, 100); // диапазон

    for (int i = 0; i < n; i++) arr[i] = d(gen); // заполняем массив
    auto t1 = chrono::high_resolution_clock::now(); // старт seq
    double a1 = average_sequential(arr, n);         // seq среднее
    auto t2 = chrono::high_resolution_clock::now(); // конец seq
    auto t3 = chrono::high_resolution_clock::now(); // старт par
    double a2 = average_parallel(arr, n);           // par среднее
    auto t4 = chrono::high_resolution_clock::now(); // конец par

    cout << "sequential average = " << a1 << "\n";  // вывод seq
    cout << "parallel average   = " << a2 << "\n";  // вывод par
    cout << "sequential time = "                      // время seq
        << chrono::duration<double, milli>(t2 - t1).count()
        << " ms\n";
    cout << "parallel time   = "                      // время par
        << chrono::duration<double, milli>(t4 - t3).count()
        << " ms\n";

    delete[] arr;                            // освобождаем память
    arr = nullptr;                           // обнуляем указатель
    return 0;                                // конец программы
}
