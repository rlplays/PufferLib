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

#include "puffer_libtorch.h"

typedef struct
{
  atomic_int work_index;
  atomic_int num_running_threads;
  volatile int num_threads;
  pthread_cond_t wake_cnd;
  pthread_t* threads;
} ThreadData;

typedef struct
{
  Env** envs;
  int num_envs;
  ThreadData* thread_data;
  struct PufferTorch* puff_torch;
  struct PufferEnvState** env_states;
} VecEnv;

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
void c_step_wrapper(Env* env, struct PufferTorch* pt, struct PufferEnvState* env_state)
{
  c_step(env);
}

void c_set_funcstep(void (*func)(Env*, struct PufferTorch*, struct PufferEnvState*))
{
  c_funcstep = func;
}

/* 

ThreadWork{ void* work_func; void* arg; int index; }
ThreadData:
 ThreadWork[N];
 work_index;
 running_index;
 cond wake, done;
 threads;
 num_threads;

c_threadstep: 
 init: running_index++ == num_threads?  signal done;
 while (1) {
   wait(wake);
    if (num_threads <= 0) break;
    running_index++;
    do {
      index = fetch_sub(work_index, 1);
      if (index >= 0) work_func(arg, index);
    } while (index > 0);
    if (running_index-- == 1) signal_done();
 }
 destroy mutex;

c_addwork ((void*)(work_func), void* arg, int index):
  if (work_index >= N) spin wait

c_vecstep:
  if (num_threads <= 0) return;
  if work_index 
  work_index = num_envs - 1;
  signal wake;

  wait done;
*/

// Main worker thread; initializes itself and runs a tight loop running through c_step (after waiting for work signal).
static void* c_threadstep(void* arg)
{
  VecEnv* vec_env = (VecEnv*)arg;

  pthread_mutex_t mtx;
  pthread_mutex_init(&mtx, NULL);
  pthread_cond_t* wake = &vec_env->thread_data->wake_cnd;

  atomic_int* work_index = &vec_env->thread_data->work_index;
  atomic_int* num_running_threads = &vec_env->thread_data->num_running_threads;
  volatile int* num_threads = &vec_env->thread_data->num_threads;
  int index;
  atomic_fetch_add(num_running_threads, 1);
  while (1)
  {
    // Wait for work
    pthread_mutex_lock(&mtx);
    pthread_cond_wait(wake, &mtx);
    pthread_mutex_unlock(&mtx);

    if (*num_threads <= 0) { break; } // Exit thread gracefully.

    // Got work to do now.
    atomic_fetch_add(num_running_threads, 1);
    do
    {
      // This is important: Go do a bunch of work in our thread, without context switches or locks
      // or any new allocs. This is the main speedup and core to ensuring the threads do as little work
      // as part of their main loop as possible. We can afford to do this as the load balancing 
      // naturally happens with mutually exclusive index values spread across threads.
      index = atomic_fetch_sub(work_index, 1);
      if (index >= 0) { c_funcstep(vec_env->envs[index], vec_env->puff_torch, vec_env->env_states[index]); }
    }
    while (index > 0);
    atomic_fetch_sub(num_running_threads, 1);
  }
  pthread_mutex_destroy(&mtx);
  return NULL;
}

//! @brief Waits for and exits all threads (if needed).
static void c_vecclose(VecEnv* vec_env)
{
  if (global_options.num_threads <= 2 || vec_env->num_envs <= 2 || !vec_env->thread_data || vec_env->thread_data->
    num_threads == 0) { return; }
  if (vec_env->thread_data->threads)
  {
    int num_threads = vec_env->thread_data->num_threads;
    atomic_store(&vec_env->thread_data->work_index, -1);
    vec_env->thread_data->num_threads = 0; // Signal to threads to exit
    pthread_cond_broadcast(&vec_env->thread_data->wake_cnd);
    // Wait for them to exit.
    while (atomic_load(&vec_env->thread_data->num_running_threads) > 0) {}

    for (int i = 0; i < num_threads; ++i)
    {
      pthread_join(vec_env->thread_data->threads[i], NULL);
    }
    pthread_cond_destroy(&vec_env->thread_data->wake_cnd);
    free(vec_env->thread_data->threads);
    vec_env->thread_data->threads = NULL;
  }
  free(vec_env->thread_data);
  if (global_options.enable_native_libtorch)
  {
    for (int i = 0; i < vec_env->num_envs; ++i)
    {
      c_freeenv(vec_env->env_states[i], vec_env->puff_torch);
    }
    free(vec_env->env_states);
    c_torch_free(vec_env->puff_torch);
    vec_env->puff_torch = NULL;
    vec_env->env_states = NULL;
  }
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
  // NOTE: On failure, we may have sem-initialized state - but it's okay because we will quit the entire program at that point.  
  vec_env->thread_data = (ThreadData*)calloc(1, sizeof(ThreadData));
  vec_env->thread_data->num_threads = global_options.num_threads;
  vec_env->thread_data->threads = (pthread_t*)calloc(vec_env->thread_data->num_threads, sizeof(pthread_t));
  if (!vec_env->thread_data->threads) { return 1; }
  if (pthread_cond_init(&vec_env->thread_data->wake_cnd, NULL) != 0) { return 1; }
  atomic_store(&vec_env->thread_data->num_running_threads, 0);
  atomic_store(&vec_env->thread_data->work_index, -1);

  for (int i = 0; i < vec_env->thread_data->num_threads; ++i)
  {
    if (pthread_create(&vec_env->thread_data->threads[i], NULL, c_threadstep, vec_env) != 0) { return 1; }
  }

  // Wait for all threads to initialize (okay to busy wait here).
  while (atomic_load(&vec_env->thread_data->num_running_threads) < vec_env->thread_data->num_threads) {}
  atomic_store_explicit(&vec_env->thread_data->num_running_threads, 0, memory_order_relaxed);
  c_set_funcstep(c_step_wrapper);
  // Must have initialized global_options via vec_enable_mt.
  if (global_options.enable_native_libtorch)
  {
    printf("Enabled native multithreading + native libtorch support with %d threads across %d envs.\n",
      vec_env->thread_data->num_threads, vec_env->num_envs);
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

//! @brief Signals worker threads to step across all environments. This is called from the main thread.
//! Returns 0 on success (1 on error).
// NOTE: Also uses the main thread to avoid having a signal/wait object.
static int c_vecstep(VecEnv* vec_env)
{
  if (vec_env->thread_data->num_threads == 0 || atomic_load(&vec_env->thread_data->work_index) >= 0) { return 1; }

  // Produce work for the worker threads.
  atomic_int* work_index = &vec_env->thread_data->work_index;
  atomic_store_explicit(work_index, vec_env->num_envs - 1, memory_order_relaxed);

  // Signal to other threads that there is new work to be done.
  pthread_cond_broadcast(&vec_env->thread_data->wake_cnd);

  // Why waste a (main) thread? (Also no need for a lock/condition variable etc).
  int index;
  do
  {
    index = atomic_fetch_sub(work_index, 1);
    if (index >= 0) { c_funcstep(vec_env->envs[index], vec_env->puff_torch, vec_env->env_states[index]); }
  }
  while (index > 0);

  // Wait for all threads to finish fully.
  // TODO(perumaal): I think this is a bad idea though - we should never spin CPU cycles busy waiting. 
  //      This is a simple initial solution and assumes SIMD-like work happening in the worker threads
  //      which significantly reduces the chance of busy waiting here.
  while (atomic_load(&vec_env->thread_data->num_running_threads) > 0) {}

  return 0;
}

