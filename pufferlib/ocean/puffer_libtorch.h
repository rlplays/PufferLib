#ifdef __cplusplus
#pragma once
#endif

struct PufferTorch;

void c_libtorch_info();
PufferTorch* c_torch_alloc(int num_actions, int obs_size);
void c_torch_free(const PufferTorch* pt);
void c_eval(const PufferTorch* pt);

