// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace fusion {
namespace cutlass_gemm_internal {
int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(GemmAllParams)>>
        &gemm_funcs,
    GemmAllParams params) {
  constexpr int WARMUP = 10;
  constexpr int REPEAT = 100;
  float min_time = 100000.f;
  int min_time_index = -1;
  for (int i = 0; i < gemm_funcs.size(); i++) {
    cutlass::Status status;
    auto func = gemm_funcs[i];
    if (!func) continue;

    for (int ii = 0; ii < WARMUP; ii++) {
      status = func(params);
    }

    cudaEvent_t beg, end;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&beg));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&end));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(beg));
    for (int ii = 0; ii < REPEAT; ii++) {
      status = func(params);
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(end));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(end));
    float elapsed_time;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventElapsedTime(&elapsed_time, beg, end));
    if (elapsed_time < min_time && status == cutlass::Status::kSuccess) {
      min_time = elapsed_time;
      min_time_index = i;
    }
  }

  if (min_time_index < 0) {
    PADDLE_THROW(
        phi::errors::NotFound("Can't find any cutlass config for this op."));
  }
  return min_time_index;
}

template <typename Destination, typename Source>
__global__ void DynamicConvert(Source const *s, Destination *t, int N) {
  cutlass::NumericConverter<Destination, Source> converter;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  CUTLASS_PRAGMA_UNROLL
  for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
    t[i] = converter(s[i]);
  }
  return;
}

template __global__ void DynamicConvert<int8_t, int32_t>(int32_t const *s,
                                                         int8_t *t,
                                                         int N);
template __global__ void DynamicConvert<int32_t, int8_t>(int8_t const *s,
                                                         int32_t *t,
                                                         int N);
template __global__ void DynamicConvert<int32_t, float>(float const *s,
                                                        int32_t *t,
                                                        int N);

template <>
__global__ void DynamicConvert<int32_t, int32_t>(int32_t const *s,
                                                 int32_t *t,
                                                 int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  CUTLASS_PRAGMA_UNROLL
  for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
    t[i] = s[i];
  }
  return;
}

template <>
__global__ void DynamicConvert<cutlass::int4b_t, int32_t>(int32_t const *s,
                                                          cutlass::int4b_t *t,
                                                          int N) {
  cutlass::NumericArrayConverter<cutlass::int4b_t, int, 8> converter;

  cutlass::Array<cutlass::int4b_t, 8> *result_ptr =
      reinterpret_cast<cutlass::Array<cutlass::int4b_t, 8> *>(t);
  cutlass::Array<int, 8> const *source_ptr =
      reinterpret_cast<cutlass::Array<int, 8> const *>(s);

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  CUTLASS_PRAGMA_UNROLL
  for (int i = idx; i < N / 8; i += gridDim.x * blockDim.x) {
    result_ptr[i] = converter(source_ptr[i]);
  }
  return;
}

__global__ void DynamicConvertWithScale(const int32_t *s,
                                        int32_t *s_extra,
                                        cutlass::int4b_t *t,
                                        int N,
                                        float scale) {
  cutlass::NumericArrayConverter<cutlass::int4b_t, int, 8> converter;

  cutlass::Array<cutlass::int4b_t, 8> *result_ptr =
      reinterpret_cast<cutlass::Array<cutlass::int4b_t, 8> *>(t);
  cutlass::Array<int, 8> *source_ptr =
      reinterpret_cast<cutlass::Array<int, 8> *>(s_extra);

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  CUTLASS_PRAGMA_UNROLL
  for (int i = idx; i < N / 8; i += gridDim.x * blockDim.x) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s_extra[i * 8 + j] =
          __float2int_rn(__fdiv_rn(__int2float_rn(s[i * 8 + j]), scale));
    }
    result_ptr[i] = converter(source_ptr[i]);
  }
  return;
}

template <typename T>
__global__ void ExpendKernel(
    const T *vector, T *matrix, const int n, const int m, const int col_major) {
  if (col_major) {
    int idx = threadIdx.x + blockIdx.x * m;
    T myval = vector[blockIdx.x % n];
    while (idx < ((blockIdx.x + 1) * m)) {
      matrix[idx] = myval;
      idx += blockDim.x;
    }
  } else {
    for (int i = 0; i < n / blockDim.x; ++i) {
      int idx = threadIdx.x + blockDim.x * i + n * blockIdx.x;
      T myval = vector[idx % n];
      while (idx < m * n) {
        matrix[idx] = myval;
        idx += n * gridDim.x;
      }
    }
    // int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // T myval = vector[idx % n];
    // while (idx < m * n) {
    //   matrix[idx] = myval;
    //   idx += gridDim.x * blockDim.x;
    // }
  }
}

template __global__ void ExpendKernel<int32_t>(const int32_t *vector,
                                               int32_t *matrix,
                                               const int n,
                                               const int m,
                                               const int col_major);
template __global__ void ExpendKernel<float>(const float *vector,
                                             float *matrix,
                                             const int n,
                                             const int m,
                                             const int col_major);
template __global__ void ExpendKernel<half>(const half *vector,
                                            half *matrix,
                                            const int n,
                                            const int m,
                                            const int col_major);

template <typename T, typename Context>
void ConvertDataToInt4(const Context &ctx,
                       const DenseTensor &source,
                       cutlass::int4b_t *output,
                       const size_t source_size,
                       const bool transpose) {
  auto stream = ctx.stream();
  DenseTensor source_prepare;
  if (transpose) {
    source_prepare = TransposeLast2Dim<T, Context>(ctx, source);
  } else {
    source_prepare = source;
  }

  constexpr int block_ = 256;
  dim3 grid((source_size + block_ - 1) / block_);
  dim3 block(block_);
  DynamicConvert<cutlass::int4b_t, T>
      <<<grid, block>>>(reinterpret_cast<const T *>(source_prepare.data()),
                        reinterpret_cast<cutlass::int4b_t *>(output),
                        source_size);
  return;
}

template void ConvertDataToInt4<int32_t, phi::GPUContext>(
    const phi::GPUContext &ctx,
    const DenseTensor &source,
    cutlass::int4b_t *output,
    const size_t source_size,
    const bool transpose);

template <typename T>
void ConvertDataToInt4(const T *source,
                       cutlass::int4b_t *output,
                       const size_t source_size,
                       cudaStream_t stream) {
  constexpr int block_ = 512;
  dim3 grid((source_size + block_ - 1) / block_);
  dim3 block(block_);
  DynamicConvert<cutlass::int4b_t, T>
      <<<grid, block>>>(source, output, source_size);
  return;
}

template void ConvertDataToInt4<int32_t>(const int32_t *source,
                                         cutlass::int4b_t *output,
                                         const size_t source_size,
                                         cudaStream_t stream);

static inline __device__ uint32_t char8_to_int4b8(int8_t a,
                                                  int8_t b,
                                                  int8_t c,
                                                  int8_t d,
                                                  int8_t e,
                                                  int8_t f,
                                                  int8_t g,
                                                  int8_t h) {
  uint32_t dst;

  uint32_t at;
  asm volatile("cvt.s32.s8 %0,%1;\n" : "=r"(at) : "h"(int16_t(a)));
  uint32_t bt;
  asm volatile("cvt.s32.s8 %0,%1;\n" : "=r"(bt) : "h"(int16_t(b)));
  uint32_t ct;
  asm volatile("cvt.s32.s8 %0,%1;\n" : "=r"(ct) : "h"(int16_t(c)));
  uint32_t dt;
  asm volatile("cvt.s32.s8 %0,%1;\n" : "=r"(dt) : "h"(int16_t(d)));
  uint32_t et;
  asm volatile("cvt.s32.s8 %0,%1;\n" : "=r"(et) : "h"(int16_t(e)));
  uint32_t ft;
  asm volatile("cvt.s32.s8 %0,%1;\n" : "=r"(ft) : "h"(int16_t(f)));
  uint32_t gt;
  asm volatile("cvt.s32.s8 %0,%1;\n" : "=r"(gt) : "h"(int16_t(g)));
  uint32_t ht;
  asm volatile("cvt.s32.s8 %0,%1;\n" : "=r"(ht) : "h"(int16_t(h)));

  asm volatile(
      "{ .reg .u32 r4;"
      "cvt.pack.sat.s4.s32.b32 r4,%8,%7,0;"
      "cvt.pack.sat.s4.s32.b32 r4,%6,%5,r4;"
      "cvt.pack.sat.s4.s32.b32 r4,%4,%3,r4;"
      "cvt.pack.sat.s4.s32.b32 %0,%2,%1,r4;"
      "}"
      : "=r"(dst)
      : "r"(at), "r"(bt), "r"(ct), "r"(dt), "r"(et), "r"(ft), "r"(gt), "r"(ht));
  return dst;
}

__global__ void convertInt8ToInt4(const int8_t *source,
                                  cutlass::int4b_t *target,
                                  const size_t source_size) {
  constexpr int32_t VPT = 16;
  int8_t *target_seg = reinterpret_cast<int8_t *>(target);
  const int32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VPT;
  int8_t source_local[VPT];
  for (int i = idx; i < source_size; i += blockDim.x * gridDim.x * VPT) {
#pragma unroll
    for (int it = 0; it < VPT / 8; ++it) {
      *reinterpret_cast<uint64_t *>(&source_local[it * 8]) =
          *reinterpret_cast<const uint64_t *>(&source[i + it * 8]);
    }
    uint32_t out_local[VPT / 8];
#pragma unroll
    for (int it = 0; it < VPT / 8; ++it) {
      out_local[it] = char8_to_int4b8(source_local[it * 8 + 0],
                                      source_local[it * 8 + 1],
                                      source_local[it * 8 + 2],
                                      source_local[it * 8 + 3],
                                      source_local[it * 8 + 4],
                                      source_local[it * 8 + 5],
                                      source_local[it * 8 + 6],
                                      source_local[it * 8 + 7]);
    }
    *reinterpret_cast<uint64_t *>(&target_seg[i / 2]) =
        *reinterpret_cast<uint64_t *>(out_local);
  }
}

template <>
void ConvertDataToInt4<int8_t>(const int8_t *source,
                               cutlass::int4b_t *output,
                               const size_t source_size,
                               cudaStream_t stream) {
  constexpr int block_ = 256;
  dim3 grid((source_size + block_ - 1) / block_);
  dim3 block(block_);
  convertInt8ToInt4<<<grid, block, 0, stream>>>(source, output, source_size);
}

// void ConvertDataToInt4WithScale(const int32_t *source,
//                                 int32_t *extra,
//                                 cutlass::int4b_t *output,
//                                 const size_t source_size,
//                                 float scale) {
//   constexpr int block_ = 512;
//   dim3 grid((source_size + block_ - 1) / block_);
//   dim3 block(block_);
//   DynamicConvertWithScale<<<grid, block>>>(
//       source, extra, output, source_size, scale);
//   return;
// }

static inline __device__ uint32_t int4_to_char4(int32_t a,
                                                int32_t b,
                                                int32_t c,
                                                int32_t d) {
  uint32_t dst;
  asm volatile("cvt.pack.sat.s8.s32.b32 %0,%1,%2,0;\n"
               : "=r"(dst)
               : "r"(d), "r"(c));
  asm volatile("cvt.pack.sat.s8.s32.b32 %0,%1,%2,%0;\n"
               : "+r"(dst)
               : "r"(b), "r"(a));
  return dst;
}

__global__ void convertInt32ToInt8(const int32_t *source,
                                   int8_t *target,
                                   const size_t size) {
  constexpr int32_t VPT = 8;
  const int32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VPT;
  int32_t source_local[VPT];
  for (int i = idx; i < size; i += blockDim.x * gridDim.x * VPT) {
#pragma unroll
    for (int it = 0; it < VPT / 2; ++it) {
      *reinterpret_cast<uint64_t *>(&source_local[it * 2]) =
          *reinterpret_cast<const uint64_t *>(&source[i + it * 2]);
    }
    uint32_t out_local[VPT / 4];
#pragma unroll
    for (int it = 0; it < VPT / 4; ++it) {
      out_local[it] = int4_to_char4(source_local[it * 4 + 0],
                                    source_local[it * 4 + 1],
                                    source_local[it * 4 + 2],
                                    source_local[it * 4 + 3]);
    }
    *reinterpret_cast<uint64_t *>(&target[i]) =
        *reinterpret_cast<uint64_t *>(out_local);
  }
}

template <typename Source, typename Target>
void ConvertData(const Source *source,
                 Target *output,
                 const size_t source_size,
                 cudaStream_t stream) {
  constexpr int block_ = 512;
  dim3 grid((source_size + block_ - 1) / block_);
  dim3 block(block_);
  DynamicConvert<Target, Source><<<grid, block>>>(source, output, source_size);
  return;
}

template <>
void ConvertData<int32_t, int8_t>(const int32_t *source,
                                  int8_t *output,
                                  const size_t source_size,
                                  cudaStream_t stream) {
  constexpr int block_ = 256;
  dim3 grid((source_size + block_ - 1) / block_);
  dim3 block(block_);
  convertInt32ToInt8<<<grid, block, 0, stream>>>(source, output, source_size);
}

template void ConvertData<int8_t, int>(const int8_t *source,
                                       int *output,
                                       const size_t source_size,
                                       cudaStream_t stream);
template void ConvertData<cutlass::half_t, int>(const cutlass::half_t *source,
                                                int *output,
                                                const size_t source_size,
                                                cudaStream_t stream);
template void ConvertData<float, int>(const float *source,
                                      int *output,
                                      const size_t source_size,
                                      cudaStream_t stream);
template void ConvertData<int, int>(const int *source,
                                    int *output,
                                    const size_t source_size,
                                    cudaStream_t stream);

}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi
