#ifdef __cplusplus
#pragma once
namespace pufferlib
{ 
#endif

struct PufferTorch;

struct PufferOptions
{
  int obs_size;
  int num_actions;
  int input_size = 128;
  int hidden_size = 128;
  bool is_multidiscrete = false;
  bool is_continuous = false;
};

void c_libtorch_info();
PufferTorch* c_torch_alloc(const PufferOptions& options);
void c_torch_free(const PufferTorch* pt);
void c_eval(const PufferTorch* pt);

#ifdef __cplusplus
}
#endif