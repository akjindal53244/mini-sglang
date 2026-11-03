#pragma once

#include <minisgl/utils.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace device {

inline constexpr auto kWarpThreads = 32u;

namespace pointer {

// Kernel row sizes and strides below are bytes, so arithmetic uses void*.
template <typename T, std::integral... U>
__always_inline __device__ auto offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<T, void>);
  return static_cast<char *>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
__always_inline __device__ auto offset(const T *ptr, U... offset) -> const
    void * {
  static_assert(std::is_same_v<T, void>);
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer
} // namespace device

namespace host {

inline void
CUDA_CHECK(cudaError_t error,
           std::source_location location = std::source_location::current()) {
  if (error != cudaSuccess)
    panic(location, "CUDA error: ", cudaGetErrorString(error));
}

inline auto get_cuda_stream(DLDevice device) -> cudaStream_t {
  // TVM FFI receives PyTorch's current stream through the DLPack environment.
  return static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
}

template <typename Kernel, typename Params>
inline void launch_kernel(Kernel kernel, std::size_t blocks,
                          std::size_t threads, DLDevice device,
                          const Params &params) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(threads);
  config.stream = get_cuda_stream(device);
  CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
}

} // namespace host
