#ifdef __cplusplus
#pragma once
#endif

#include "puffer_nativecpp.h"

// puffer_nativecpp.cpp is compiled as a separate unit so we need to glue these here. (Env is not visible outside the env's binding.c).
// TODO: Env should really be a well-defined struct in its own header instead of #define'd inside the env ?
float* get_obs_ptr(Env* env) { return env->observations; }
int* get_actions_ptr(Env* env) { return env->actions; }
float* get_rewards_ptr(Env* env) { return env->rewards; }
unsigned char* get_terminals_ptr(Env* env) { return env->terminals; }

void c_single_step(void* vec_env, int index) { c_step(((VecEnv*)vec_env)->envs[index]); }
