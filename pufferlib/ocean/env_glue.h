#ifdef __cplusplus
#pragma once
#endif

#include "puffer_native_eval.h"

// These glue methods helps env_binding use these methods from the C side while the new native
// puffer_native_eval.cpp is compiled as a separate unit in C++ land. (Env is not visible outside the env's binding.c).
#ifdef __cplusplus
extern "C"
{
#endif

void c_add_to_log(VecEnv* envs, Env* env, int env_index) 
{
  // Maintain separate aggregate log per env to avoid locking.
  Log* aggregate = &envs->aggregate_log[env_index];
  const int num_keys = sizeof(Log) / sizeof(float);
  for (int j = 0; j < num_keys; j++) {
      ((float*)aggregate)[j] += ((float*)&env->log)[j];
      ((float*)&env->log)[j] = 0.0f;
  }
}

#ifdef PUFFERLIB_SELFPLAY

// Implement these in your env and use  `SELF_PLAY=1 python setup.py build_<ext>` to enable self-play weight transfer support in the training code. 
// This is optional and only needed if you want to do self-play with native libtorch eval. 
// You do need to implement the self-play yourself - really depends on your env needs/setup.

// #ifdef PUFFERLIB_SELFPLAY
//! @brief Whether we should transfer weights from the training model to envs for self-play. 
//!        This is checked every epoch, so the training code can toggle this on/off as needed 
//!        (e.g. only transfer every N epochs or if syllabus changed, etc. as it's an expensive operation).
// inline bool c_should_transfer_selfplay_weights();

//! @brief Transfers the LSTM weights if c_should_transfer_selfplay_weights() returns true. (TODO: Rename if there are other models).
// inline void c_transfer_selfplay_weights(Env* env, int env_index,
//   float* encoder_w, int encoder_w_size, float* encoder_b, int encoder_b_size,
//   float* decoder_w, int decoder_w_size, float* decoder_b, int decoder_b_size,
//   float* value_w, int value_w_size, float* value_b, int value_b_size,
//   float* weight_ih, int weight_ih_size, float* weight_hh, int weight_hh_size,
//   float* bias_ih, int bias_ih_size, float* bias_hh, int bias_hh_size);
// #endif

bool extern_should_transfer_selfplay_weights(VecEnv* vec_env)
{
  return c_should_transfer_selfplay_weights();
}

void extern_transfer_selfplay_weights(VecEnv* vec_env, int env_index,
  float* encoder_w, int encoder_w_size, float* encoder_b, int encoder_b_size,
  float* decoder_w, int decoder_w_size, float* decoder_b, int decoder_b_size,
  float* value_w, int value_w_size, float* value_b, int value_b_size,
  float* weight_ih, int weight_ih_size, float* weight_hh, int weight_hh_size,
  float* bias_ih, int bias_ih_size, float* bias_hh, int bias_hh_size)
{
  c_transfer_selfplay_weights(vec_env->envs[env_index], env_index,
    encoder_w, encoder_w_size,
    encoder_b, encoder_b_size,
    decoder_w, decoder_w_size,
    decoder_b, decoder_b_size,
    value_w, value_w_size,
    value_b, value_b_size,
    weight_ih, weight_ih_size,
    weight_hh, weight_hh_size,
    bias_ih, bias_ih_size,
    bias_hh, bias_hh_size);
}

#endif



// TODO(perumaal): These must be static inlined so the tight inner loop avoids multiple lea/call overheads.
// This requires a redesign of Env to be a proper struct knowable in advance rather than a #define macro hack.
// For now, this isn't a concern as the env step is way more expensive for envs we care about than these pointer fetches.

// The C++ code needs a glue to call this as an extern "C" function in case the binding is also itself a C++ code. A mess.
//! @brief Steps a single env that's part of a batch (called from multithreaded puffer_native).
void c_step_batch(void* arg, int env_index, int env_batch_local_index, int32_t* actions_data, int num_actions,
  float* rewards, float* terminals, int horizon_segment)
{
  VecEnv* vec_env =(VecEnv*)arg;
  Env* env = vec_env->envs[env_index];
  c_add_to_log(vec_env, env, env_index);

  // Clear rewards/terminals before each step (in case the env author forgot to).
  env->rewards[0] = 0;
  env->terminals[0] = 0;

  // Fill actions, step and send rewards/terminals back.
  int32_t* actions = &actions_data[env_batch_local_index * num_actions];
  for (int i = 0; i < num_actions; i++)
  {
    // We assume discrete actions; will be cast to the appropriate action type.
    env->actions[i] = (int)actions[i];
  }
  c_step(env);
  
  // Doing rewards/terminals+clipping here also maintains cache locality as the env step just wrote to these pointers.
  // Note the rewards/terminals/obs are copied with the next segment.
  float r = env->rewards[0];
  r = (r < -1.0f ? -1.0f : (r > 1.0f ? 1.0f : r));
  rewards[env_batch_local_index] = r;
  terminals[env_batch_local_index] = (env->terminals[0] > 0.5f ? 1.0f : 0.0f);

  // obs automatically transfers via memory-mapped pointers to obs tensors.
}

void c_single_step(void* envs, int index) { c_step(((Env**)envs)[index]); }

// sizeof(Log) is unknown outside of env_glue.h (i.e. in puffer_native.*). So this glue function helps set it up.
void c_setup_log(VecEnv* vec_env) 
{
  // Log is float-only, so zeroing it out is safe this way.
  memset(vec_env->aggregate_log, 0, sizeof(Log) * vec_env->num_envs);
}

#ifdef __cplusplus
}
#endif

//! @brief Inits vectorized multi-threading envs with provided num threads. Returns 0 on success (1 on error).
static int c_vecinit(struct VecEnv* vec_env)
{
  // If we have only a couple envs, it's not worth parallelizing. Also, don't penalize the user as they
  // may want to change the .ini dynamically without having to worry about this.
  if (vec_env->opts.num_threads_env == 0 || vec_env->num_envs <= 2)
  {
    vec_env->opts.num_threads_env = 0;
    return 1;
  }
  c_init_multithreading(vec_env);
  if (vec_env->opts.enable_native_libtorch)
  {
    vec_env->puff_torch = c_torch_alloc(vec_env);
    vec_env->aggregate_log = (Log*) calloc(vec_env->num_envs, sizeof(Log));
  }
  else
  {
    vec_env->aggregate_log = NULL;
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

  if (vec_env->aggregate_log)
  {
    free(vec_env->aggregate_log);
    vec_env->aggregate_log = NULL;
  }
}

//! @brief Old multithreaded step function for vec envs without native libtorch support.
//! Returns 0 on success (1 on error).
static int c_vecstep(struct VecEnv* vec_env)
{
  c_start_work(vec_env);
  c_add_work_batched(vec_env, c_single_step, vec_env->envs, 0, vec_env->num_envs - 1);
  c_wait_all_done(vec_env);
  return 0;
}
