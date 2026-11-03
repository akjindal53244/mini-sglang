#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct IndexKernelParams {
  void *__restrict__ output;
  const void *__restrict__ weight;
  const void *__restrict__ indices;
  std::size_t num_warps;
};

struct MaskedKernelParams {
  IndexKernelParams params;
  std::size_t start;
  std::size_t length;
};

template <std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(128, 1) void index_kernel(
    const __grid_constant__ IndexKernelParams params) {
  using namespace device;
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = 128 / kWarpThreads;

  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits];
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    const auto src = pointer::offset(weight, pos * kSize,
                                     (warp_id % kNumSplits) * kSizePerWarp);
    warp::copy<kSizePerWarp>(dst, src);
  }
}

template <std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(128, 1) void masked_index_kernel(
    const __grid_constant__ MaskedKernelParams mask_params) {
  using namespace device;
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = 128 / kWarpThreads;

  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[params, start, length] = mask_params;
  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits] - start;
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    if (pos < length) {
      const auto src = pointer::offset(weight, pos * kSize,
                                       (warp_id % kNumSplits) * kSizePerWarp);
      warp::copy<kSizePerWarp>(dst, src);
    } else {
      warp::reset<kSizePerWarp>(dst);
    }
  }
}

// Each row is split across num_splits warps; each block has four warps.
template <std::size_t element_size, std::size_t num_splits = 1>
void launch_index(const tvm::ffi::TensorView weights,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView output,
                  tvm::ffi::Optional<tvm::ffi::Tuple<int, int>> mask_opts) {
  using namespace host;
  const auto device = weights.device();
  check_cuda_tensor(weights, 2, device);
  check_cuda_tensor(output, 2, device);
  check_indices(indices, device);
  const auto num_indices = indices.size(0);
  RuntimeCheck(output.size(0) == num_indices &&
                   output.size(1) == weights.size(1),
               "Embedding output shape must be [num_indices, embedding_dim]");
  RuntimeCheck(output.dtype() == weights.dtype(), "Embedding dtype mismatch");
  const auto entry_size = dtype_bytes(weights.dtype()) * weights.size(1);
  RuntimeCheck(entry_size == element_size, "Embedding row byte size mismatch");
  if (num_indices == 0)
    return;
  const auto use_int32 = indices.dtype().bits == 32;
  constexpr auto num_threads = 128;

  constexpr auto kWarpPerBlock = num_threads / 32;
  const auto num_warps = num_splits * num_indices;
  const auto num_blocks = div_ceil(num_warps, kWarpPerBlock);
  const auto params = IndexKernelParams{
      .output = static_cast<char *>(output.data_ptr()),
      .weight = static_cast<const char *>(weights.data_ptr()),
      .indices = indices.data_ptr(),
      .num_warps = num_warps,
  };

  if (mask_opts.has_value()) {
    const auto &obj = mask_opts.value();
    const auto [start, length] = obj;
    RuntimeCheck(start >= 0 && length >= 0 && length <= weights.size(0),
                 "Invalid vocabulary shard range");
    const auto m_params = MaskedKernelParams{
        .params = params,
        .start = static_cast<std::size_t>(start),
        .length = static_cast<std::size_t>(length),
    };
    const auto kernel =
        use_int32 ? masked_index_kernel<element_size, num_splits, int32_t>
                  : masked_index_kernel<element_size, num_splits, int64_t>;
    launch_kernel(kernel, num_blocks, num_threads, device, m_params);
  } else {
    const auto kernel = use_int32
                            ? index_kernel<element_size, num_splits, int32_t>
                            : index_kernel<element_size, num_splits, int64_t>;
    launch_kernel(kernel, num_blocks, num_threads, device, params);
  }
}

} // namespace
