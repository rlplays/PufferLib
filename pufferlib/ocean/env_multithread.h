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


float* get_obs_ptr(Env* env) { return env->observations; }
int* get_actions_ptr(Env* env) { return env->actions; }
float* get_rewards_ptr(Env* env) { return env->rewards; }
unsigned char* get_terminals_ptr(Env* env) { return env->terminals; }


//! @brief Waits for and exits all threads (if needed).
static void c_vecclose(struct VecEnv* vec_env)
{
  c_shutdown_multithreading(vec_env);
  for (int i = 0; i < vec_env->num_envs; ++i)
  {
    c_freeenv(vec_env->env_states[i], vec_env->puff_torch);
  }

  if (vec_env->env_states)
  {
    free(vec_env->env_states);
    vec_env->env_states = NULL;
  }
  if (vec_env->puff_torch)
  {
    c_torch_free(vec_env->puff_torch);
    vec_env->puff_torch = NULL;
  }
}

//! @brief Inits multi-threading with provided num threads. Returns 0 on success (1 on error).
//! NOTE: Must set {@related global_options.num_threads} before calling this function.
static int c_multithread_init(struct VecEnv* vec_env)
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
    vec_env->puff_torch = c_torch_alloc(&global_options, vec_env);
    vec_env->env_states = (struct PufferEnvState**)calloc(vec_env->num_envs, sizeof(struct PufferEnvState*));
    for (int i = 0; i < vec_env->num_envs; ++i)
    {
      vec_env->env_states[i] = c_initenv(vec_env->puff_torch, i);
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
static int c_vecstep(struct VecEnv* vec_env)
{
  if (global_options.enable_native_libtorch)
  {
    // Must use the c_native_fulleval instead that does action (inference) + step segmented across a BPTT horizon.
    return 1;
  }
  c_start_work(vec_env);
  c_add_work_batched(vec_env, c_single_step, vec_env, 0, vec_env->num_envs - 1);
  c_wait_all_done(vec_env);
  return 0;
}
