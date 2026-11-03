#pragma once

#include <minisgl/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>

namespace host {

// Validate metadata on CPU; never copy tensor contents back from the GPU.
inline void check_cuda_tensor(tvm::ffi::TensorView tensor, int ndim,
                              DLDevice device, bool contiguous = true) {
  RuntimeCheck(tensor.ndim() == ndim, "Unexpected tensor rank");
  RuntimeCheck(tensor.device().device_type == kDLCUDA &&
                   tensor.device().device_id == device.device_id,
               "Tensors must share a CUDA device");
  RuntimeCheck(!contiguous || tensor.is_contiguous(),
               "Tensor must be contiguous");
}

inline void check_indices(tvm::ffi::TensorView indices, DLDevice device) {
  check_cuda_tensor(indices, 1, device);
  const auto dtype = indices.dtype();
  RuntimeCheck(dtype.code == kDLInt && dtype.lanes == 1 &&
                   (dtype.bits == 32 || dtype.bits == 64),
               "Indices must be int32 or int64");
}

} // namespace host
