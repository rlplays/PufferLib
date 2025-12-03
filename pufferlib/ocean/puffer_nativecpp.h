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

struct ThreadData;

struct VecEnv
{
  Env** envs;
  int num_envs;
  ThreadData* thread_data; // remove.
  struct Threading* threading;
  struct PufferTorch* puff_torch;
  // Per-env state for the LSTM model.
  struct PufferEnvState** env_states;
};


// Initialize using c_setup_pufferoptions (no constructor/defaults in C :()
//! @brief Options for vec envs' puffer torch LSTM model.
struct PufferOptions
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
};

#define DEFAULT_INPUT_SIZE (128)
#define DEFAULT_HIDDEN_SIZE (128)

#if defined(__cplusplus) 
extern "C"
{
#endif


// Setup and cleanup of PufferOptions.
void c_setup_pufferoptions(PufferOptions* options, int num_actions, int num_logits, int input_size,
  int hidden_size, bool is_continuous);
void c_cleanup_pufferoptions(PufferOptions* options);

// Overall initialization across all envs.
void c_libtorch_info();
struct PufferTorch* c_torch_alloc(PufferOptions* options);
void c_torch_load_weights(struct PufferTorch* pt, struct Weights* weights);
void c_torch_free(struct PufferTorch* pt);

// Per-env state+eval. 
// Update weights and init once per env for a single horizon.
struct PufferEnvState* c_initenv(struct PufferTorch* pt);
void c_evalenv(struct PufferEnvState* state, struct PufferTorch* pt, float* obs, int* actions);
void c_freeenv(struct PufferEnvState* state, struct PufferTorch* pt);

// Threading support
void c_init_multithreading(PufferOptions* options, VecEnv* vec_env);
void c_shutdown_multithreading(VecEnv* vec_env);
typedef void (*work_func)(void* arg, int index);
void c_add_work(VecEnv* vec_env, work_func func, void* arg, int index);
void c_wait_all_done(VecEnv* vec_env);

#if defined(__cplusplus)
}
#endif

#endif
