#ifdef __cplusplus
#pragma once
#endif

#include "puffer_nativecpp.h"

// These glue methods helps env_binding use these methods from the C side while the new native
// puffer_nativecpp.cpp is compiled as a separate unit in C++ land. (Env is not visible outside the env's binding.c).
#ifdef __cplusplus
extern "C"
{
#endif
// TODO(perumaal): These must be static inlined so the tight inner loop avoids multiple lea/call overheads.
// This requires a redesign of Env to be a proper struct knowable in advance rather than a #define macro hack.
// For now, this isn't a concern as the env step is way more expensive for envs we care about than these pointer fetches.
float* get_obs_ptr(Env* env) { return env->observations; }
int* get_actions_ptr(Env* env) { return env->actions; }
float* get_rewards_ptr(Env* env) { return env->rewards; }
unsigned char* get_terminals_ptr(Env* env) { return env->terminals; }

// The C++ code needs a glue to call this as an extern "C" function in case the binding is also itself a C++ code. A mess.
void c_step_glue(Env* env) { c_step(env); }
void c_single_step(void* vec_env, int index) { c_step(((VecEnv*)vec_env)->envs[index]); }
#ifdef __cplusplus
}
#endif

//! @brief Inits vectorized multi-threading envs with provided num threads. Returns 0 on success (1 on error).
static int c_vecinit(struct VecEnv* vec_env)
{
  // If we have only a couple envs, it's not worth parallelizing. Also, don't penalize the user as they
  // may want to change the .ini dynamically without having to worry about this.
  if (vec_env->opts.num_threads == 0 || vec_env->num_envs <= 2)
  {
    vec_env->opts.num_threads = 0;
    return 1;
  }
  c_init_multithreading(vec_env);
  if (vec_env->opts.enable_native_libtorch)
  {
    vec_env->puff_torch = c_torch_alloc(vec_env);
  }
  else
  {
    vec_env->puff_torch = NULL;
  }
  return 0;
}

//! @brief Waits for and exits all threads (if needed).
static void c_vecclose(struct VecEnv* vec_env)
{
  c_shutdown_multithreading(vec_env);

  if (vec_env->puff_torch)
  {
    c_torch_free(vec_env->puff_torch);
    vec_env->puff_torch = NULL;
  }
}

//! @brief Old multithreaded step function for vec envs without native libtorch support.
//! Returns 0 on success (1 on error).
static int c_vecstep(struct VecEnv* vec_env)
{
  if (vec_env->opts.enable_native_libtorch)
  {
    // Must use the c_native_fulleval instead that does action (inference) + step segmented across a BPTT horizon.
    return 1;
  }
  c_start_work(vec_env);
  c_add_work_batched(vec_env, c_single_step, vec_env, 0, vec_env->num_envs - 1);
  c_wait_all_done(vec_env);
  return 0;
}
