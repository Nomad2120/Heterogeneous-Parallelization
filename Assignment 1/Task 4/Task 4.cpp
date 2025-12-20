#include <iostream> // ввод вывод
#include <random> // рандом
#include <chrono> // время
#include <omp.h> // openmp
using namespace std; // чтобы не писать std::

int main() { // начало программы
    int n = 5000000; // размер массива
    int* arr = new int[n]; // динамически выделяем массив
    random_device rd; // источник
    mt19937 gen(rd()); // генератор
    uniform_int_distribution<int> d(1, 100); // диапазон 1..100

    for (int i = 0; i < n; i++) arr[i] = d(gen); // заполняем массив
    auto s1 = chrono::high_resolution_clock::now(); // старт seq
    long long sum_seq = 0; // сумма seq
    for (int i = 0; i < n; i++) sum_seq += arr[i]; // считаем сумму seq
    double avg_seq = (double)sum_seq / n; // среднее seq
    auto e1 = chrono::high_resolution_clock::now(); // конец seq
    double t_seq = chrono::duration<double, milli>(e1 - s1).count(); // время seq
    auto s2 = chrono::high_resolution_clock::now(); // старт par

    long long sum_par = 0; // сумма par
#pragma omp parallel for reduction(+:sum_par) // параллельная сумма
    for (int i = 0; i < n; i++) sum_par += arr[i]; // складываем
    double avg_par = (double)sum_par / n; // среднее par

    auto e2 = chrono::high_resolution_clock::now(); // конец par
    double t_par = chrono::duration<double, milli>(e2 - s2).count(); // время par

    cout << "seq average = " << avg_seq << "\n"; // вывод seq
    cout << "par average = " << avg_par << "\n"; // вывод par
    cout << "seq time = " << t_seq << " ms\n"; // время seq
    cout << "par time = " << t_par << " ms\n"; // время par

    delete[] arr; // освобождаем память
    arr = nullptr; // обнуляем указатель
    return 0; // конец программы
}
