#ifdef __cplusplus
#pragma once
#include <cstdlib>
#include <cassert>
#include <cstdint>
#else
#include <stdlib.h>
#include <assert.h>
#endif
#ifndef PUFFER_LIBTORCH_H
#define PUFFER_LIBTORCH_H
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

// Merge with vec
// Initialize using c_setup_pufferoptions (no constructor/defaults in C :()
typedef struct PufferOptions
{
  int obs_size;
  int num_actions;
  int input_size;
  int hidden_size;
  bool is_continuous;
  // Will be alloc'ed by c_setup_pufferoptions.
  int64_t* logit_sizes;
  // For multidiscrete only: total number of action logits.
  int num_atns;
  int num_threads;
} PufferOptions;

// Setup and cleanup of PufferOptions.
void c_setup_pufferoptions(PufferOptions* options, int num_logits);
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
#endif