#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct StoreKernelParams {
  void *__restrict__ k_cache;
  void *__restrict__ v_cache;
  const void *__restrict__ indices;
  const void *__restrict__ k;
  const void *__restrict__ v;
  std::size_t kv_cache_stride;
  std::size_t kv_input_stride;
  std::size_t length;
};

template <std::size_t kElementSize, std::integral T>
__global__ __launch_bounds__(128, 1) void store_kv_cache(
    const __grid_constant__ StoreKernelParams params) {
  using namespace device;

  constexpr auto kWarpPerBlock = 128 / kWarpThreads;

  const auto &[k_cache, v_cache, indices, k, v, kv_cache_stride,
               kv_input_stride, length] = params;
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  // One warp copies a K row and its V row; gaps between rows are allowed.
  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride);
    const auto src_k = pointer::offset(k, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_k, src_k);
    const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride);
    const auto src_v = pointer::offset(v, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_v, src_v);
  }
}

template <std::size_t element_size>
void launch_store(const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView k, const tvm::ffi::TensorView v) {
  using namespace host;
  const auto device = k_cache.device();
  check_cuda_tensor(k_cache, 2, device, false);
  check_cuda_tensor(v_cache, 2, device, false);
  check_cuda_tensor(k, 2, device, false);
  check_cuda_tensor(v, 2, device, false);
  check_indices(indices, device);
  const auto length = static_cast<std::size_t>(indices.size(0));
  const auto width = k_cache.size(1);
  RuntimeCheck(v_cache.size(0) == k_cache.size(0) && v_cache.size(1) == width,
               "K/V cache shapes differ");
  RuntimeCheck(k.size(0) == length && v.size(0) == length &&
                   k.size(1) == width && v.size(1) == width,
               "K/V input shape must be [num_indices, cache_width]");
  for (const auto tensor : {k_cache, v_cache, k, v}) {
    RuntimeCheck(tensor.dtype() == k_cache.dtype(), "K/V dtype mismatch");
    RuntimeCheck(width == 1 || tensor.stride(1) == 1,
                 "K/V rows must be contiguous");
  }
  // QKV slices and interleaved cache tensors can have gaps between rows.
  // Each K/V pair shares its row stride; the input and cache strides can
  // differ.
  RuntimeCheck(length <= 1 || k.stride(0) == v.stride(0),
               "K/V input strides differ");
  RuntimeCheck(k_cache.size(0) <= 1 || k_cache.stride(0) == v_cache.stride(0),
               "K/V cache strides differ");
  const auto dtype_size = dtype_bytes(k_cache.dtype());
  RuntimeCheck(element_size == dtype_size * width,
               "K/V row byte size mismatch");
  if (length == 0)
    return;
  const auto use_int32 = indices.dtype().bits == 32;
  const auto kv_cache_stride = k_cache.stride(0) * dtype_size;
  const auto kv_input_stride = k.stride(0) * dtype_size;
  constexpr auto num_threads = 128;

  const auto params = StoreKernelParams{
      .k_cache = k_cache.data_ptr(),
      .v_cache = v_cache.data_ptr(),
      .indices = indices.data_ptr(),
      .k = k.data_ptr(),
      .v = v.data_ptr(),
      .kv_cache_stride = kv_cache_stride,
      .kv_input_stride = kv_input_stride,
      .length = length,
  };

  constexpr auto kWarpPerBlock = num_threads / 32;
  static_assert(num_threads % 32 == 0);
  const auto num_blocks = div_ceil(length, kWarpPerBlock);
  const auto kernel = use_int32 ? store_kv_cache<element_size, int32_t>
                                : store_kv_cache<element_size, int64_t>;
  launch_kernel(kernel, num_blocks, num_threads, device, params);
}

} // namespace
