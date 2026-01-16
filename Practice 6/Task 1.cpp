%%writefile task1_opencl_add.cpp
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h> // opencl
#include <iostream> // вывод
#include <vector> // массивы
#include <fstream> // чтение .cl
#include <cstdlib> // rand
#include <ctime> // time
#include <chrono> // таймер
#include <cmath> // fabs

using namespace std; // чтоб не писать std::

void cl_ok(cl_int err, const char* msg) { // проверка ошибок opencl
    if (err != CL_SUCCESS) { // если ошибка
        cout << "ошибка opencl (" << msg << "): " << err << "\n"; // вывод
        exit(1); // выход
    }
}

string read_text(const char* path) { // читаем файл в строку
    ifstream f(path); // открываем
    if (!f.is_open()) { // если не открылся
        cout << "не могу открыть файл: " << path << "\n"; // вывод
        exit(1); // выход
    }
    string s((istreambuf_iterator<char>(f)), istreambuf_iterator<char>()); // читаем всё
    return s; // возвращаем
}

bool get_device(cl_device_type type, cl_platform_id& out_plat, cl_device_id& out_dev) { // ищем устройство
    cl_uint pcount = 0; // кол-во платформ
    cl_int err = clGetPlatformIDs(0, nullptr, &pcount); // сколько платформ
    if (err != CL_SUCCESS || pcount == 0) return false; // нет платформ
    vector<cl_platform_id> plats(pcount); // список платформ
    cl_ok(clGetPlatformIDs(pcount, plats.data(), nullptr), "get platforms"); // получаем

    for (cl_uint pi = 0; pi < pcount; pi++) { // перебор платформ
        cl_uint dcount = 0; // кол-во устройств
        err = clGetDeviceIDs(plats[pi], type, 0, nullptr, &dcount); // сколько устройств
        if (err != CL_SUCCESS || dcount == 0) continue; // нет
        vector<cl_device_id> devs(dcount); // список устройств
        cl_ok(clGetDeviceIDs(plats[pi], type, dcount, devs.data(), nullptr), "get devices"); // получаем
        out_plat = plats[pi]; // платформа
        out_dev = devs[0]; // первое устройство
        return true; // нашли
    }
    return false; // не нашли
}

double run_cpu_plain(const vector<float>& A, const vector<float>& B, vector<float>& C) { // обычный cpu (без opencl)
    auto s = chrono::high_resolution_clock::now(); // старт
    int n = (int)A.size(); // n
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i]; // сложение
    auto e = chrono::high_resolution_clock::now(); // конец
    return chrono::duration<double, milli>(e - s).count(); // мс
}

double run_gpu_opencl(const vector<float>& A, const vector<float>& B, vector<float>& C) { // opencl gpu
    cl_platform_id plat = nullptr; // платформа
    cl_device_id dev = nullptr; // устройство
    if (!get_device(CL_DEVICE_TYPE_GPU, plat, dev)) return -1.0; // нет gpu

    cl_int err = 0; // ошибки
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); // контекст
    cl_ok(err, "create context"); // ok

    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); // очередь
    cl_ok(err, "create queue"); // ok

    string src = read_text("kernel_add.cl"); // читаем ядро
    const char* csrc = src.c_str(); // c-string
    size_t srclen = src.size(); // длина

    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err); // программа
    cl_ok(err, "create program"); // ok

    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr); // build
    if (err != CL_SUCCESS) { // если не собралась
        size_t log_size = 0; // лог
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size); // размер
        vector<char> log(log_size); // буфер
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr); // читаем
        cout << "build log:\n" << log.data() << "\n"; // вывод
        cl_ok(err, "build program"); // ошибка
    }

    cl_kernel ker = clCreateKernel(prog, "vector_add", &err); // kernel
    cl_ok(err, "create kernel"); // ok

    int n = (int)A.size(); // n
    size_t bytes = (size_t)n * sizeof(float); // bytes

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err); // A
    cl_ok(err, "buffer A"); // ok
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err); // B
    cl_ok(err, "buffer B"); // ok
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err); // C
    cl_ok(err, "buffer C"); // ok

    cl_ok(clEnqueueWriteBuffer(q, dA, CL_TRUE, 0, bytes, A.data(), 0, nullptr, nullptr), "write A"); // H2D
    cl_ok(clEnqueueWriteBuffer(q, dB, CL_TRUE, 0, bytes, B.data(), 0, nullptr, nullptr), "write B"); // H2D

    cl_ok(clSetKernelArg(ker, 0, sizeof(cl_mem), &dA), "arg0"); // A
    cl_ok(clSetKernelArg(ker, 1, sizeof(cl_mem), &dB), "arg1"); // B
    cl_ok(clSetKernelArg(ker, 2, sizeof(cl_mem), &dC), "arg2"); // C
    cl_ok(clSetKernelArg(ker, 3, sizeof(int), &n), "arg3"); // n

    size_t global = (size_t)n; // global
    size_t local = 256; // local
    if (global % local != 0) global = ((global / local) + 1) * local; // округляем

    cl_event ev = nullptr; // event
    cl_ok(clEnqueueNDRangeKernel(q, ker, 1, nullptr, &global, &local, 0, nullptr, &ev), "run kernel"); // запуск
    cl_ok(clFinish(q), "finish"); // ждём

    cl_ok(clEnqueueReadBuffer(q, dC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr), "read C"); // D2H

    cl_ulong t0 = 0; // start
    cl_ulong t1 = 0; // end
    cl_ok(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, nullptr), "profile start"); // start
    cl_ok(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t1, nullptr), "profile end"); // end

    double ms = (double)(t1 - t0) / 1e6; // ns -> ms

    clReleaseEvent(ev); // free
    clReleaseMemObject(dA); // free
    clReleaseMemObject(dB); // free
    clReleaseMemObject(dC); // free
    clReleaseKernel(ker); // free
    clReleaseProgram(prog); // free
    clReleaseCommandQueue(q); // free
    clReleaseContext(ctx); // free

    return ms; // ms
}

bool check_add(const vector<float>& A, const vector<float>& B, const vector<float>& C) { // проверка
    int n = (int)A.size(); // n
    for (int i = 0; i < n; i++) { // цикл
        float ref = A[i] + B[i]; // эталон
        if (fabs(C[i] - ref) > 1e-5f) return false; // если ошибка
    }
    return true; // ок
}

int main() { // main
    srand((unsigned)time(0)); // seed
    int sizes[3] = {10000, 100000, 1000000}; // размеры

    cout << "n,cpu_ms,gpu_ms,ok_cpu,ok_gpu\n"; // csv

    for (int t = 0; t < 3; t++) { // цикл
        int n = sizes[t]; // n
        vector<float> A(n); // A
        vector<float> B(n); // B
        vector<float> Ccpu(n); // C cpu
        vector<float> Cgpu(n); // C gpu

        for (int i = 0; i < n; i++) { // fill
            A[i] = (float)(rand() % 1000) / 10.0f; // A
            B[i] = (float)(rand() % 1000) / 10.0f; // B
        }

        double cpu_ms = run_cpu_plain(A, B, Ccpu); // cpu plain
        double gpu_ms = run_gpu_opencl(A, B, Cgpu); // gpu opencl

        int ok_cpu = check_add(A, B, Ccpu) ? 1 : 0; // ok cpu
        int ok_gpu = (gpu_ms < 0) ? 0 : (check_add(A, B, Cgpu) ? 1 : 0); // ok gpu

        cout << n << "," << cpu_ms << "," << gpu_ms << "," << ok_cpu << "," << ok_gpu << "\n"; // csv строка
    }

    return 0; // конец
}
