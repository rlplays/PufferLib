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
#define PUFFER_ASSERT(cond, msg)                      \
  do {                                                \
    if (!(cond)) {                                    \
      fprintf(stderr, "Assertion failed: %s\n", msg); \
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
struct PufferEnvState;
struct Env;

typedef struct VecEnv
{
  Env** envs;
  int num_envs;
  struct Threading* threading;
  struct PufferTorch* puff_torch;
  // Per-env state for the LSTM model.
  struct PufferEnvState** env_states;
} VecEnv;


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
  // Will be alloc'ed by c_setup_pufferoptions.
  int64_t* logit_sizes;
  // For multidiscrete only: total number of action logits.
  int num_atns;
  int num_threads;
  // TODO(perumaal): Merge all of this with env_multithread stuff (vec env state?).
} PufferOptions;

#define DEFAULT_INPUT_SIZE (128)
#define DEFAULT_HIDDEN_SIZE (128)

#if defined(__cplusplus)
extern "C"
{
#endif


// Setup and cleanup of PufferOptions.
void c_setup_pufferoptions(struct PufferOptions* options, int num_actions, int num_logits, int input_size,
  int hidden_size, bool is_continuous);
void c_cleanup_pufferoptions(struct PufferOptions* options);


// TODO(perumaal): Cleanup this header to only have strict C-compatible stuff here. Everything else is isolated to the C++ impl.
// Overall initialization across all envs (mainly for testing purposes).
void c_libtorch_info();
void c_torch_load_weights(struct PufferTorch* pt, struct Weights* weights);

// Manage torch state and obtain the puffer torch instance for use later.
struct PufferTorch* c_torch_alloc(struct PufferOptions* options);
void c_torch_free(struct PufferTorch* pt);

// Per-env state+eval (this is pre-batch code; not used by the batch stuff). 
// Update weights and init once per env for a single horizon.
struct PufferEnvState* c_initenv(struct PufferTorch* pt, int env_index);
void c_evalenv(struct PufferEnvState* state, struct PufferTorch* pt, float* obs, int* actions);
void c_freeenv(struct PufferEnvState* state, struct PufferTorch* pt);

// Threading support (for both the internal libtorch's native multithreading and the existing C 
// native multithreading glued with the C++ threading impl).
// These are generic threading support and have no direct dependency on libtorch or any particular impl itself.

//! @brief Initializes T threads (in options) for M envs (in vec_env).
void c_init_multithreading(struct PufferOptions* options, struct VecEnv* vec_env);

//! @brief Waits for all threads to finish, join them all and exit.
void c_shutdown_multithreading(struct VecEnv* vec_env);

//! @brief Work item func that takes a void* arg and an index that was provided at the queueing time.
typedef void (*work_func)(void* arg, int index);

//! @brief Async queues up a work item to be executed by one of the threads. 
//! NOTE: The work must be meaningful enough (chunky) as this is lock-based and a bit more expensive than pure atomics).
void c_add_work(struct VecEnv* vec_env, work_func func, void* arg, int index);

//! @brief Waits for all queued work to be done.
void c_wait_all_done(struct VecEnv* vec_env);

#if defined(__cplusplus)
}
#endif

#endif
