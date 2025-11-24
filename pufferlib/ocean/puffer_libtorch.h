#ifdef __cplusplus
#pragma once
#include <cstdlib>
#else
#include <stdlib.h>
#endif

#include "puffernet.h"

#ifdef __cplusplus
namespace pufferlib
{
#endif

// Internal C interface that hides C++ stuff internally and is the only thing needed for the API.
struct PufferTorch;

struct PufferOptions
{
  int obs_size;
  int num_actions;
  int* logit_sizes = nullptr;
  int input_size = 128;
  int hidden_size = 128;
  bool is_multidiscrete = false;
  bool is_continuous = false;
};
// Setup and cleanup of PufferOptions.
void c_setup_pufferoptions(PufferOptions* options, int num_logits);
void c_cleanup_pufferoptions(PufferOptions* options);

void c_libtorch_info();
PufferTorch* c_torch_alloc(PufferOptions* options);
void c_torch_load_weights(PufferTorch* pt, Weights* weights);
void c_torch_free(const PufferTorch* pt);
void c_eval(const PufferTorch* pt);

#ifdef __cplusplus
}
#endif
