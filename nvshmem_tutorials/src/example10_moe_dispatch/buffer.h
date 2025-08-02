#ifndef KATO_BUFFER_H
#define KATO_BUFFER_H

#include <cerrno>
#include <cstdint>
#include <sys/types.h>
#include <vector>
#include <cuda_runtime.h>

namespace kato {

    template <typename T> struct DeviceBuffer {
        uint32_t size;
        uint32_t element_size;
        T *data = nullptr;

        DeviceBuffer(const std::vector<T> &a) {
            size = a.size();
            element_size = sizeof(T);

            cudaMalloc(&data, size * sizeof(T));
            cudaMemcpy(data, a.data(), size * sizeof(T), cudaMemcpyHostToDevice);
        }

        const T *get() const { return data; }
        T *get() { return data; }

        uint32_t getSize() const { return size; }

        uint32_t getElementSize() const { return element_size; }

        ~DeviceBuffer() { cudaFree(data); }
    };

} // namespace kato

#endif