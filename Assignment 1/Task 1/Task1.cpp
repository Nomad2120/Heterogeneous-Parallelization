#include <iostream> // ввод вывод
#include <random>   // рандом
using namespace std; // чтобы не писать std::

int main() { // начало программы
    int n = 50000; // размер массива
    int* arr = new int[n]; // динамически выделяем массив
    random_device rd; // источник
    mt19937 gen(rd()); // генератор
    uniform_int_distribution<int> d(1, 100); // диапазон 1..100

    for (int i = 0; i < n; i++) arr[i] = d(gen); // заполняем массив
    long long sum = 0; // сумма элементов

    for (int i = 0; i < n; i++) sum += arr[i]; // считаем сумму
    double average = (double)sum / n; // считаем среднее значение
    cout << "average value = " << average << "\n"; // вывод результата

    delete[] arr; // освобождаем память
    arr = nullptr; // обнуляем указатель
    return 0; // конец программы
}
