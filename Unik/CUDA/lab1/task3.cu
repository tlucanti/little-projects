

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

#include "common.hpp"

#define N (512 * 1024)

int main(void)
{
    thrust::host_vector<int> a(N);
    thrust::host_vector<int> b(N);
    thrust::host_vector<int> c(N);
    
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = i * i;
    }
    
    thrust::device_vector<int> ca(a);
    thrust::device_vector<int> cb(b);
    thrust::device_vector<int> cc(c);
    
    timer_start();
    thrust::transform(ca.begin(), cb.begin(), cb.begin(), cc.begin(), thrust::plus<int>());
    float t = timer_stop();

    for (int i = 0; i < min(N, 10); i++) {
        std::cout << ca[i] << ' ' << cb[i] << ' ' << cc[i] << '\n';
    }
    std::cout << "execution time: " << t << '\n';
}
