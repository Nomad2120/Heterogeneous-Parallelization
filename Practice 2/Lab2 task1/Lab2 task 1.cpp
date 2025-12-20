#include <iostream> // ввод и вывод
#include <random> // рандом
using namespace std; // чтобы не писать std::

// сортировка пузырьком
void bubble_sort(int* arr, int n) { // массив и размер
    for (int i = 0; i < n - 1; i++) { // внешний цикл
        for (int j = 0; j < n - i - 1; j++) { // внутренний цикл
            if (arr[j] > arr[j + 1]) { // если элементы не по порядку
                int tmp = arr[j]; // сохраняем первый
                arr[j] = arr[j + 1]; // меняем
                arr[j + 1] = tmp; // меняем
            }
        }
    }
}

// сортировка выборкой
void selection_sort(int* arr, int n) { // массив и размер
    for (int i = 0; i < n - 1; i++) { // идем по массиву
        int min_index = i; // индекс минимума
        for (int j = i + 1; j < n; j++) { // ищем минимум
            if (arr[j] < arr[min_index]) { // если нашли меньше
                min_index = j; // запоминаем индекс
            }
        }
        int tmp = arr[i]; // меняем местами
        arr[i] = arr[min_index]; // первый элемент
        arr[min_index] = tmp; // минимум
    }
}

// сортировка вставкой
void insertion_sort(int* arr, int n) { // массив и размер
    for (int i = 1; i < n; i++) { // начинаем со второго
        int key = arr[i]; // текущий элемент
        int j = i - 1; // индекс слева
        while (j >= 0 && arr[j] > key) { // пока больше key
            arr[j + 1] = arr[j]; // сдвигаем вправо
            j--; // идем влево
        }
        arr[j + 1] = key; // вставляем элемент
    }
}

void print_array(int* arr, int n) { // функция печати
    for (int i = 0; i < n; i++) // цикл
        cout << arr[i] << " "; // вывод элемента
    cout << "\n"; // новая строка
}

int main() { // начало программы
    int n; // размер массива
    cout << "array size: "; // сообщение
    cin >> n; // ввод размера
    if (n <= 0) return 1; // проверка
    int* arr = new int[n]; // динамический массив
    random_device rd; // источник рандома
    mt19937 gen(rd()); // генератор
    uniform_int_distribution<int> d(1, 100); // диапазон
    for (int i = 0; i < n; i++) // заполняем массив
        arr[i] = d(gen); // случайное число
    cout << "original array:\n"; // вывод текста
    print_array(arr, n); // печать массива

    // выбор сортировки, какая нужно убираем из комментов
    // bubble_sort(arr, n); // пузырек
    // selection_sort(arr, n); // выборкой
    insertion_sort(arr, n); // вставкой

    cout << "sorted array:\n"; // вывод текста
    print_array(arr, n); // печать массива
    delete[] arr; // освобождаем память
    arr = nullptr; // обнуляем указатель
    return 0; // конец программы
}
