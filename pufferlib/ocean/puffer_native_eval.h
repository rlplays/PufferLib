#ifdef __cplusplus
#pragma once
#include <cstdlib>
#include <cassert>
#include <cstdint>

#include <puffer_cuda.h>

#define DELETE_ARRAY(ptr) \
  do {                 \
    delete[] ptr;     \
    ptr = nullptr;    \
  }                                                                                                                    \
  while (0)
#define DELETE_PTR(ptr) \
  do {                 \
    delete ptr;      \
    ptr = nullptr;   \
  }                                                                                                                   \
  while (0)
#else
#include <stdlib.h>
#include <assert.h>
#endif

#ifndef PUFFER_NATIVE_EVAL_H
#define PUFFER_NATIVE_EVAL_H
#if defined(DEBUG)

inline static void PUFFER_ASSERT_BREAK()
{
#if defined(_MSC_VER)
  // assert (abort) does not break into the debugger in VS 2022 ! It's insane, so we have to use this weird contraption that's cross platform.
  __debugbreak();
#elif defined(__clang__) || defined(__GNUC__)
  __builtin_trap();
#else
  /* Fallback method */
  *((volatile int*)0) = 0; /* This will cause a segmentation fault */
#endif
}

#define PUFFER_ASSERT(cond, msg)                      \
  do {                                                \
    if (!(cond)) {                                    \
      fprintf(stderr, "Assertion failed: %s\n", msg); \
      PUFFER_ASSERT_BREAK();                          \
      assert(cond);                                   \
    }                                                 \
  } while (0)
#else
#define PUFFER_ASSERT(cond, msg) ((void)0)
#endif

#include <stdint.h>

// Global forward declaration(s).
struct Weights;

// Internal C interface that hides C++ stuff internally and is the only thing needed for the API.
struct PufferTorch;
struct Env;
struct Log;



#ifndef PUFFER_EXTERN
// The main env_binding header is included in both C and C++ files (and from binding.c from each env). 
// Which means in C++, we have to access the c_step_batch with C linkage, but in C code, it's just a normal function.
struct Env;
struct VecEnv;
#define PUFFER_EXTERN extern "C"
#endif

// Initialize using c_setup_pufferoptions (no constructor/defaults in C :()
//! @brief Options for vec envs' puffer torch LSTM model.
typedef struct PufferOptions
{
  //! @brief Whether to enable the whole libtorch functionality natively for eval.
  bool enable_native_libtorch;
  //! @brief Whether to enable the whole libtorch functionality natively for train.
  bool enable_native_libtorch_train;
  int obs_size;
  int num_actions;
  int num_logits;
  //! @brief LSTM(i) tensor size.
  int input_size;
  //! @brief LSTM(h) tensor size.
  int hidden_size;
  bool is_continuous;
  // Will be alloc'ed by c_setup_pufferoptions.
  int64_t* logit_sizes;
  // For multidiscrete only: total number of action logits.
  int num_atns;
  int num_threads_env;
  int num_threads_batch;
  int num_gpu_batches;
  int bptt_horizon;
} PufferOptions;

typedef struct VecEnv
{
  Env** envs;
  int num_envs;
  struct Threading* threading_batch;
  struct Threading* threading_env;
  struct PufferTorch* puff_torch;
  struct PufferOptions opts;
  struct Log* aggregate_log;
} VecEnv;

#define DEFAULT_INPUT_SIZE (128)
#define DEFAULT_HIDDEN_SIZE (128)


#if defined(__cplusplus)
extern "C"
{
#endif

// Setup and cleanup PufferOptions with logits array.
void c_setup_pufferoptions(struct VecEnv* vec_env, int num_actions, int num_logits, int input_size,
  int hidden_size, bool is_continuous, int num_gpu_batches);
void c_cleanup_pufferoptions(struct VecEnv* vec_env);

// Manage torch state and obtain the puffer torch instance for use later.
struct PufferTorch* c_torch_alloc(struct VecEnv* vec_env);
void c_torch_free(struct PufferTorch* pt);

// Threading support (for both the internal libtorch's native multithreading and the existing C 
// native multithreading glued with the C++ threading impl).
// These are generic threading support and have no direct dependency on libtorch or any particular impl itself.

//! @brief Initializes T threads (in options) for M envs (in vec_env).
void c_init_multithreading(struct VecEnv* vec_env);

//! @brief Waits for all threads to finish, join them all and exit.
void c_shutdown_multithreading(struct VecEnv* vec_env);

//! @brief Work item func that takes a void* arg and an index that was provided at the queueing time.
typedef void (*work_func)(void* arg, int index);

//! @brief Start overall work (verify there is nothing in the queue to start off).
void c_start_work(struct VecEnv* vec_env);

//! @brief Async queues up a batched work item to be sharded across multiple threads.
//! Calls func(arg, index) for each index in [start_index, end_index] i.e. inclusive indices.
void c_add_work_batched(struct VecEnv* vec_env, work_func func, void* arg, int start_index, int end_index);

//! @brief Waits for all queued work to be done.
void c_wait_all_done(struct VecEnv* vec_env);

#if defined(__cplusplus)

bool assign_training_weights(VecEnv* vec_env, torch::Tensor& encoder_linear_w, torch::Tensor& encoder_linear_b,
  torch::Tensor& decoder_linear_w, torch::Tensor& decoder_linear_b, torch::Tensor& value_w, torch::Tensor& value_b,
  torch::Tensor& weight_ih, torch::Tensor& weight_hh, torch::Tensor& bias_ih, torch::Tensor& bias_hh,
  Tensor encoder_linear_w_in, torch::Tensor encoder_linear_b_in,
  torch::Tensor decoder_linear_w_in, torch::Tensor decoder_linear_b_in,
  torch::Tensor value_w_in, torch::Tensor value_b_in,
  torch::Tensor weight_ih_in, torch::Tensor weight_hh_in,
  torch::Tensor bias_ih_in, torch::Tensor bias_hh_in);
}
#endif

#endif // PUFFER_NATIVE_EVAL_H

