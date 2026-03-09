#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

typedef unsigned long long ull;
const ull iter = 1000000000;
const int num_threads = std::thread::hardware_concurrency();
std::mutex lock;

ull one_thread_sum() {
    auto start = std::chrono::high_resolution_clock::now();
    ull sum = 0;
    for (ull i = 1; i <= iter; i++) {
        sum += i;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    std::cout << "\n1 thread time: " << dur.count() << " ms\nresult: " << sum << "\n";
    return sum;
}
void th_worker(ull frst, ull last, ull& save) {
    ull sum = 0;
    for (ull i = frst + 1; i <= last; ++i) {
        sum += i;
    }
    {
        std::lock_guard<std::mutex> lock_guard(lock);
        save += sum;
    }
}
ull all_threads_sum() {
    ull step = iter / num_threads;
    ull sum = 0;
    std::vector<std::thread> ths;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        ull rem = (i == num_threads - 1) ? iter % num_threads : 0;
        ths.emplace_back(th_worker, i * step, (i + 1) * step + rem, std::ref(sum));
    }

    for (auto& th : ths) {
        th.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    std::cout << num_threads << " threads time: " << dur.count() << " ms\nresult: " << sum << "\n";
    return sum;
}
int main() {
    for (int i = 0; i < 10; ++i) {
        std::cout << "====== Ex. " << i << " ======\n";
        one_thread_sum();
        all_threads_sum();
        std::cout << '\n';
    }
    return 0;
}
