#include <musa_runtime.h>
#include <murand_kernel.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

template <typename T>
__device__ __forceinline__ T MurandUniformHalfOpen01(
    murandStatePhilox4_32_10_t* state);

template <>
__device__ __forceinline__ float MurandUniformHalfOpen01<float>(
    murandStatePhilox4_32_10_t* state) {
  return 1.0f - murand_uniform(state);
}

template <>
__device__ __forceinline__ double MurandUniformHalfOpen01<double>(
    murandStatePhilox4_32_10_t* state) {
  return 1.0 - murand_uniform_double(state);
}

template <typename T>
__device__ __forceinline__ T MurandStandardNormal(
    murandStatePhilox4_32_10_t* state);

template <>
__device__ __forceinline__ float MurandStandardNormal<float>(
    murandStatePhilox4_32_10_t* state) {
  return murand_normal(state);
}

template <>
__device__ __forceinline__ double MurandStandardNormal<double>(
    murandStatePhilox4_32_10_t* state) {
  return murand_normal_double(state);
}

template <typename T>
__global__ void RandomUniformKernel(int64_t n, uint64_t seed, uint64_t seed2,
                                    T* output) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  murandStatePhilox4_32_10_t state;
  murand_init(static_cast<unsigned long long>(seed),
              static_cast<unsigned long long>(seed2),
              static_cast<unsigned long long>(idx), &state);
  output[idx] = MurandUniformHalfOpen01<T>(&state);
}

template <typename T>
__global__ void RandomStandardNormalKernel(int64_t n, uint64_t seed,
                                           uint64_t seed2, T* output) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  murandStatePhilox4_32_10_t state;
  murand_init(static_cast<unsigned long long>(seed),
              static_cast<unsigned long long>(seed2),
              static_cast<unsigned long long>(idx), &state);
  output[idx] = MurandStandardNormal<T>(&state);
}

template <typename T>
__global__ void TruncatedNormalKernel(int64_t n, uint64_t seed, uint64_t seed2,
                                      T* output) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  murandStatePhilox4_32_10_t state;
  murand_init(static_cast<unsigned long long>(seed),
              static_cast<unsigned long long>(seed2),
              static_cast<unsigned long long>(idx), &state);

  T value = static_cast<T>(0);
  for (int attempt = 0; attempt < 100; ++attempt) {
    const T sample = MurandStandardNormal<T>(&state);
    if (fabs(sample) <= static_cast<T>(2.0)) {
      value = sample;
      break;
    }
  }
  output[idx] = value;
}

template <typename T>
__device__ __forceinline__ typename std::make_unsigned<T>::type NextUnsigned(
    murandStatePhilox4_32_10_t* state);

template <>
__device__ __forceinline__ uint32_t NextUnsigned<int>(
    murandStatePhilox4_32_10_t* state) {
  return murand(state);
}

template <>
__device__ __forceinline__ uint64_t NextUnsigned<int64_t>(
    murandStatePhilox4_32_10_t* state) {
  const uint64_t hi = static_cast<uint64_t>(murand(state));
  const uint64_t lo = static_cast<uint64_t>(murand(state));
  return (hi << 32) | lo;
}

template <typename T>
__global__ void RandomUniformIntKernel(int64_t n, uint64_t seed, uint64_t seed2,
                                       T minval, T maxval, T* output) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  using U = typename std::make_unsigned<T>::type;

  murandStatePhilox4_32_10_t state;
  murand_init(static_cast<unsigned long long>(seed),
              static_cast<unsigned long long>(seed2),
              static_cast<unsigned long long>(idx), &state);

  const U min_u = static_cast<U>(minval);
  const U max_u = static_cast<U>(maxval);
  const U range = max_u - min_u;

  U random_u = 0;
  if (range == 0) {
    output[idx] = minval;
    return;
  }

  const U max_random = std::numeric_limits<U>::max();
  const U limit = max_random - (max_random % range);
  do {
    random_u = NextUnsigned<T>(&state);
  } while (random_u >= limit);

  output[idx] = static_cast<T>(min_u + (random_u % range));
}

inline int GetBlocks(int64_t n, int threads) {
  return static_cast<int>((n + threads - 1) / threads);
}

#define DEFINE_LAUNCH_UNIFORM(T)                                                \
  void LaunchRandomUniform_##T(void* stream, int64_t n, uint64_t seed_raw,      \
                               T* output) {                                      \
    int threads = 256;                                                           \
    int blocks = GetBlocks(n, threads);                                          \
    RandomUniformKernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(       \
        n, seed_raw, 0ULL, output);                                              \
  }

#define DEFINE_LAUNCH_NORMAL(T)                                                 \
  void LaunchRandomStandardNormal_##T(void* stream, int64_t n, uint64_t seed_raw,\
                                      T* output) {                              \
    int threads = 256;                                                          \
    int blocks = GetBlocks(n, threads);                                         \
    RandomStandardNormalKernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(\
        n, seed_raw, 0ULL, output);                                             \
  }

#define DEFINE_LAUNCH_TRUNCATED(T)                                              \
  void LaunchTruncatedNormal_##T(void* stream, int64_t n, uint64_t seed_raw,    \
                                 T* output) {                                    \
    int threads = 256;                                                           \
    int blocks = GetBlocks(n, threads);                                          \
    TruncatedNormalKernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(     \
        n, seed_raw, 0ULL, output);                                              \
  }

#define DEFINE_LAUNCH_INT(T)                                                    \
  void LaunchRandomUniformInt_##T(void* stream, int64_t n, uint64_t seed_raw,   \
                                  T minval, T maxval, T* output) {               \
    int threads = 256;                                                           \
    int blocks = GetBlocks(n, threads);                                          \
    RandomUniformIntKernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(    \
        n, seed_raw, 0ULL, minval, maxval, output);                              \
  }

DEFINE_LAUNCH_UNIFORM(float)
DEFINE_LAUNCH_UNIFORM(double)

DEFINE_LAUNCH_NORMAL(float)
DEFINE_LAUNCH_NORMAL(double)

DEFINE_LAUNCH_TRUNCATED(float)
DEFINE_LAUNCH_TRUNCATED(double)

DEFINE_LAUNCH_INT(int)
DEFINE_LAUNCH_INT(int64_t)
