#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>

int main() {
    std::cout << "Thrust " << THRUST_MAJOR_VERSION << "."
              << THRUST_MINOR_VERSION << "." << THRUST_SUBMINOR_VERSION
              << " | CUDA " << CUDART_VERSION << "\n";

    const int n = 1000000;
    thrust::device_vector<int> d(n);
    thrust::sequence(d.begin(), d.end(), 1);

    long long sum = thrust::reduce(d.begin(), d.end(), 0LL);
    std::cout << "sum(1.." << n << ") = " << sum << "\n";

    thrust::device_vector<long long> squares(n);
    thrust::transform(d.begin(), d.end(), squares.begin(),
                      thrust::square<long long>());
    std::cout << "last square = " << squares.back() << "\n";

    thrust::sort(d.rbegin(), d.rend());
    std::cout << "sorted descending, first = " << d.front()
              << ", last = " << d.back() << "\n";

    bool ok = (sum == static_cast<long long>(n) * (n + 1) / 2);
    std::cout << (ok ? "OK" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
