#ifdef __cplusplus
#pragma once
#include <cstdlib>
#include <cassert>
#else
#include <stdlib.h>
#include <assert.h>
#endif

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

#include "puffernet.h"

#ifdef __cplusplus
namespace pufferlib
{
#endif

// Internal C interface that hides C++ stuff internally and is the only thing needed for the API.
struct PufferTorch;
struct PufferEnvState;

struct PufferOptions
{
  int obs_size;
  int num_actions;
  int input_size = 128;
  int hidden_size = 128;
  bool is_multidiscrete = false;
  bool is_continuous = false;
  // Will be filled in by the libtorch code.
  int* logit_sizes = nullptr;
  // For multidiscrete only: total number of action logits.
  int num_atns = 0;
};

// Setup and cleanup of PufferOptions.
void c_setup_pufferoptions(PufferOptions* options, int num_logits);
void c_cleanup_pufferoptions(PufferOptions* options);

// Overall initialization across all envs.
void c_libtorch_info();
PufferTorch* c_torch_alloc(PufferOptions* options);
void c_torch_load_weights(PufferTorch* pt, Weights* weights);
void c_torch_free(PufferTorch* pt);

// Per-env state+eval.
PufferEnvState* c_initenv(PufferTorch* pt);
void c_evalenv(PufferEnvState* state, PufferTorch* pt, float* obs, int* actions);
void c_freeenv(PufferEnvState* state, PufferTorch* pt);
#ifdef __cplusplus
}
#endif
