#ifndef SHIP_BUFFER_H
#define SHIP_BUFFER_H

#include <cerrno>
#include <cstdint>
#include <sys/types.h>
#include <vector>
#include <cuda_runtime.h>

namespace ship {

    template <typename T> struct DeviceBuffer {
        uint32_t size;
        T *data = nullptr;

        DeviceBuffer(const std::vector<T> &a) {
            size = a.size();
            cudaMalloc(&data, size * sizeof(T));
            cudaMemcpy(data, a.data(), size * sizeof(T), cudaMemcpyHostToDevice);
        }

        const T *get() const { return data; }
        T *get() { return data; }

        ~DeviceBuffer() { cudaFree(data); }
    };

} // namespace ship

#endif