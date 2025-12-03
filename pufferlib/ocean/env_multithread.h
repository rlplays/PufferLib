#ifdef __cplusplus
#pragma once
#include <atomic>
#include <thread>
#include <condition_variable>
#include <mutex>

#else
#include <stdatomic.h>
#include <pthread.h>
#endif

#include "puffer_nativecpp.h"


static struct PufferOptions global_options = {0};
static void (*c_funcstep)(Env*, struct PufferTorch*, struct PufferEnvState*) = NULL;

struct PufferEnvState* get_envstate(VecEnv* vec_env, int env_index)
{
  if (!vec_env->env_states) { return NULL; }
  return vec_env->env_states[env_index];
}

struct PufferTorch* get_puffertorch(VecEnv* vec_env) { return vec_env->puff_torch; }

int get_numenvstates(VecEnv* vec_env)
{
  if (!vec_env->env_states) { return 0; }
  return vec_env->num_envs;
}

float* get_obs_ptr(Env* env) { return env->observations; }
int* get_actions_ptr(Env* env) { return env->actions; }
float* get_rewards_ptr(Env* env) { return env->rewards; }
unsigned char* get_terminals_ptr(Env* env) { return env->terminals; }


//! @brief Waits for and exits all threads (if needed).
static void c_vecclose(VecEnv* vec_env)
{
  c_shutdown_multithreading(vec_env);
}

//! @brief Inits multi-threading with provided num threads. Returns 0 on success (1 on error).
//! NOTE: Must set {@related global_options.num_threads} before calling this function.
static int c_multithread_init(VecEnv* vec_env)
{
  // If we have only a couple envs, it's not worth parallelizing. Also, don't penalize the user as they
  // may want to change the .ini dynamically without having to worry about this.
  if (global_options.num_threads <= 2 || vec_env->num_envs <= 2)
  {
    global_options.num_threads = 0;
    return 1;
  }
  c_init_multithreading(&global_options, vec_env);
  // Must have initialized global_options via vec_enable_mt.
  if (global_options.enable_native_libtorch)
  {
    printf("Enabled native multithreading + native libtorch support with %d threads across %d envs.\n",
      global_options.num_threads, vec_env->num_envs);
    vec_env->puff_torch = c_torch_alloc(&global_options);
    vec_env->env_states = (struct PufferEnvState**)calloc(vec_env->num_envs, sizeof(struct PufferEnvState*));
    for (int i = 0; i < vec_env->num_envs; ++i)
    {
      vec_env->env_states[i] = c_initenv(vec_env->puff_torch);
      if (!vec_env->env_states[i]) { return 1; }
    }
  }
  else
  {
    vec_env->puff_torch = NULL;
    vec_env->env_states = NULL;
  }
  return 0;
}

void c_single_step(void* vec_env, int index) { c_step(((VecEnv*)vec_env)->envs[index]); }

//! @brief Old multithreaded step function for vec envs without native libtorch support.
//! Returns 0 on success (1 on error).
// NOTE: Also uses the main thread to avoid having a signal/wait object.
static int c_vecstep(VecEnv* vec_env)
{
  for (int i = 0; i < vec_env->num_envs; ++i)
  {
    Env* env = vec_env->envs[i];
    c_add_work(vec_env, c_single_step, vec_env, i);
  }
  return 0;
}

