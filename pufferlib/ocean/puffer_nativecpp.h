#ifdef __cplusplus
#pragma once
#include <cstdlib>
#include <cassert>
#include <cstdint>
#else
#include <stdlib.h>
#include <assert.h>
#endif


#ifndef PUFFER_NATIVECPP_H
#define PUFFER_NATIVECPP_H
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

// Initialize using c_setup_pufferoptions (no constructor/defaults in C :()
//! @brief Options for vec envs' puffer torch LSTM model.
typedef struct PufferOptions
{
  //! @brief Whether to enable the whole libtorch functionality natively.
  bool enable_native_libtorch;
  int obs_size;
  int num_actions;
  int num_logits;
  //! @brief LSTM(i) tensor size.
  int input_size;
  //! @brief LSTM(h) tensor size.
  int hidden_size;
  bool is_continuous;
  int batch_chunk_size_kb; // in KiB
  // Will be alloc'ed by c_setup_pufferoptions.
  int64_t* logit_sizes;
  // For multidiscrete only: total number of action logits.
  int num_atns;
  int num_threads;
  int bptt_horizon;
} PufferOptions;

typedef struct VecEnv
{
  Env** envs;
  int num_envs;
  struct Threading* threading;
  struct PufferTorch* puff_torch;
  struct PufferOptions opts;
} VecEnv;

#define DEFAULT_INPUT_SIZE (128)
#define DEFAULT_HIDDEN_SIZE (128)

#if defined(__cplusplus)
extern "C"
{
#endif

// Setup and cleanup PufferOptions with logits array.
void c_setup_pufferoptions(struct VecEnv* vec_env, int num_actions, int num_logits, int input_size,
  int hidden_size, bool is_continuous, int batch_chunk_size_kb);
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


//! @brief Inits vectorized multi-threading envs with provided num threads. Returns 0 on success (1 on error).
static int c_vecinit(struct VecEnv* vec_env)
{
  // If we have only a couple envs, it's not worth parallelizing. Also, don't penalize the user as they
  // may want to change the .ini dynamically without having to worry about this.
  if (vec_env->opts.num_threads <= 2 || vec_env->num_envs <= 2)
  {
    vec_env->opts.num_threads = 0;
    return 1;
  }
  c_init_multithreading(vec_env);
  if (vec_env->opts.enable_native_libtorch)
  {
    vec_env->puff_torch = c_torch_alloc(vec_env);
  }
  else
  {
    vec_env->puff_torch = NULL;
  }
  return 0;
}

//! @brief Waits for and exits all threads (if needed).
static void c_vecclose(struct VecEnv* vec_env)
{
  c_shutdown_multithreading(vec_env);

  if (vec_env->puff_torch)
  {
    c_torch_free(vec_env->puff_torch);
    vec_env->puff_torch = NULL;
  }
}

void c_single_step(void* vec_env, int index);

//! @brief Old multithreaded step function for vec envs without native libtorch support.
//! Returns 0 on success (1 on error).
static int c_vecstep(struct VecEnv* vec_env)
{
  if (vec_env->opts.enable_native_libtorch)
  {
    // Must use the c_native_fulleval instead that does action (inference) + step segmented across a BPTT horizon.
    return 1;
  }
  c_start_work(vec_env);
  c_add_work_batched(vec_env, c_single_step, vec_env, 0, vec_env->num_envs - 1);
  c_wait_all_done(vec_env);
  return 0;
}

#if defined(__cplusplus)
}
#endif

#endif // PUFFER_NATIVECPP_H

