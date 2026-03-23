#pragma once
// Threading support for puffer_nativecpp.
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <torch/torch.h>

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

// Have to manually pass the actions_data so each env can choose to decipher actions (for e.g. breakout uses float* for discrete actions).
PUFFER_EXTERN void c_step_batch(void* arg, int env_index, int env_batch_local_index, void* actions_data,
  int num_actions, float* rewards, float* terminals, int step_count);

PUFFER_EXTERN void c_setup_log(VecEnv* vec_env);

#ifdef PUFFERLIB_SELFPLAY
PUFFER_EXTERN bool extern_should_transfer_selfplay_weights(VecEnv* vec_env);
PUFFER_EXTERN void extern_transfer_selfplay_weights(VecEnv* vec_env, int env_index,
  float* encoder_w, int encoder_w_size, float* encoder_b, int encoder_b_size,
  float* decoder_w, int decoder_w_size, float* decoder_b, int decoder_b_size,
  float* value_w, int value_w_size, float* value_b, int value_b_size,
  float* weight_ih, int weight_ih_size, float* weight_hh, int weight_hh_size,
  float* bias_ih, int bias_ih_size, float* bias_hh, int bias_hh_size);



#endif

// Optional completion function that will be called back after all the batch tasks are completed.
struct BatchCompletion
{
  //! @brief Called when {@ref done_tasks} equals {@ref batch_total_tasks}. Called at most once per batch.
  std::function<void(void*)> batch_completion_cb;
  std::atomic_int done_tasks = 0;
  std::atomic_int batch_total_tasks = 0;

  BatchCompletion(const std::function<void(void*)>& batch_completion) : batch_completion_cb(batch_completion)
  {
    PUFFER_ASSERT(batch_completion != nullptr, "BatchGroup requires a non-empty callback.");
  }

  explicit BatchCompletion() = delete; // Do not allow passing in an empty callback.
};

//
// Threading support.
//

struct WorkBatch
{
  std::function<void(void*, int)> func;
  void* arg;
  //! @brief Inclusive start index of a batch. Each thread wakes up processes ~batch_size work items
  //! in the range [start_index, end_index]. Once all the work in this range is processed, the optional
  //! batch completion callback is called and this work batch is relinquished.
  int start_index;
  //! @brief Inclusive end index of a batch. Does not change during processing.
  int end_index;
  //! @brief Number of items to process in one go (the actual total starting work item count is end_index-start_index+1).
  int batch_size;
  // Using a shared_ptr here to avoid locks (so the last thread that goes out of scope automatically releases this).
  // Also prevents alloc'ing completion stuff when there is no need to. Tried using a raw ptr here first, but it's
  // tricky to get right with multi-threading, would have reinvented shared_ptr anyways.
  std::shared_ptr<BatchCompletion> batch_completion;
};

inline void set_current_thread_high_priority()
{
#ifdef _WIN32
  // Have to move this into a separate compile unit to avoid Windows.h inclusion issues.
  //HANDLE process = GetCurrentProcess();
  //SetPriorityClass(process, HIGH_PRIORITY_CLASS);
  //HANDLE thread = GetCurrentThread();
  //SetThreadPriority(thread, THREAD_PRIORITY_ABOVE_NORMAL);
#else
  // TODO(perumaal): Must set thread affinity to pin to specific cores?
  errno = 0;
  int old_nice = nice(0);
  if (old_nice != -1 || errno == 0) { old_nice = nice(-1); }
  else
  {
    sched_param sch_params{};
    // TODO(perumaal): Probably shouldn't use the max priority? 
    sch_params.sched_priority = sched_get_priority_max(SCHED_OTHER);
    pthread_t this_thread = pthread_self();
    pthread_setschedparam(this_thread, SCHED_OTHER, &sch_params);
  }
#endif
}

void c_thread_func(void* arg);

struct Threading
{
  std::deque<WorkBatch> work_batches;
  std::vector<std::thread> threads;
  std::atomic_int num_threads;
  std::mutex work_mutex;
  std::condition_variable work_cv;
  std::condition_variable done_cv;
  std::atomic_int batch_count{0};

  explicit Threading(const int num_threads) : num_threads(num_threads)
  {
    //work_batches.reserve(work_capacity);
    for (int i = 0; i < num_threads; i++)
    {
      threads.emplace_back(std::thread([this] { this->c_thread_func(); }));
    }
  }

  // Wait for signal to do work, do work, signal if there is no more work in the queue.
  inline void c_thread_func()
  {
    set_current_thread_high_priority();
    int last_count = 0;
    int end_index = 0;
    while (true)
    {
      WorkBatch work;
      {
        std::unique_lock lock(work_mutex);
        // This ensures that wait_all_done is guaranteed to not miss a done_cv notification.
        if (last_count == 1) { done_cv.notify_all(); }
        while (!(num_threads.load() == 0 || !work_batches.empty())) { work_cv.wait(lock); }
        // Shortcuts to exit or try again in case we got woken up but no work.
        if (num_threads.load() == 0) { break; }
        if (work_batches.empty()) { continue; }
        work = work_batches.front();
        end_index = work.start_index + work.batch_size - 1;
        if (work.start_index >= work.end_index || end_index + (work.batch_size / 2) >= work.end_index)
        {
          end_index = work.end_index;
          work_batches.pop_front(); // We have reserved space, so this won't realloc.
        }
        else
        {
          work_batches.front().start_index = end_index + 1;
        }
        batch_count.fetch_add(1);
      } // lock scope

      const auto start_index = work.start_index;
      const auto& func = work.func;
      auto* arg = work.arg;
      // Tight loop that runs the batch's envs in the [start_index, end_index] range.
      for (int i = start_index; i <= end_index; i++)
      {
        func(arg, i);
      }

      // Check for completion before we go back to holding a lock/waiting for further work.
      auto completion = work.batch_completion;
      if (completion != nullptr)
      {
        // Must store `done` locally (this avoids a lock).
        const auto completed_count = end_index - start_index + 1;
        const auto done = work.batch_completion->done_tasks.fetch_add(completed_count) + completed_count;
        if (done == completion->batch_total_tasks) { completion->batch_completion_cb(work.arg); }
      }
      work = {}; // Relinquish any captured closures.
      last_count = batch_count.fetch_sub(1);
    }
  }

  void wait_all_done()
  {
    std::unique_lock<std::mutex> lock(work_mutex);
    // This ensures that any in-progress work items finish fully before we return.
    while (batch_count.load() != 0 || !work_batches.empty()) { done_cv.wait(lock); }
  }

  void add_work(const WorkBatch& work)
  {
    if (num_threads.load() == 0)
    {
      PUFFER_ASSERT(false, "Must have created at least one thread to add work to.");
      return;
    }
    {
      std::lock_guard<std::mutex> lock(work_mutex);
      work_batches.push_back(work);
    }
    work_cv.notify_one();
  }

  ~Threading()
  {
    num_threads.store(0);
    work_cv.notify_all();
    wait_all_done();
    for (auto& thread : threads)
    {
      if (thread.joinable())
      {
        thread.join();
      }
    }
    threads.clear();
  }

  void check_empty()
  {
    std::lock_guard<std::mutex> lock(work_mutex);
    PUFFER_ASSERT(work_batches.empty() && batch_count.load() == 0, "Work queue not empty at start of work.");
  }
};

void c_init_multithreading(VecEnv* vec_env)
{
  PufferOptions* options = &vec_env->opts;
  PUFFER_ASSERT(options != nullptr && options->num_threads_env > 0 && vec_env->threading_env == nullptr,
    "Invalid options/thread data.");
  PUFFER_ASSERT(options != nullptr && options->num_threads_batch > 0 && vec_env->threading_batch == nullptr,
    "Invalid options/thread data.");
  vec_env->threading_env = new Threading(options->num_threads_env);
  vec_env->threading_batch = new Threading(options->num_threads_batch);
}

void c_shutdown_multithreading(VecEnv* vec_env)
{
  if (vec_env->threading_env != nullptr && vec_env->threading_batch != nullptr)
  {
    c_wait_all_done(vec_env);
    DELETE_PTR(vec_env->threading_env);
    DELETE_PTR(vec_env->threading_batch);
  }
}

void c_start_work(struct VecEnv* vec_env)
{
  PUFFER_ASSERT(vec_env->threading_env != nullptr && vec_env->threading_batch != nullptr, "Invalid threading state.");
  vec_env->threading_batch->check_empty();
  vec_env->threading_env->check_empty();
}


enum class PufferWorkType
{
  //! @brief Raw env stepping work that is CPU bound and should not be blocked by GPU work that 
  //!        is independent of CPU / system RAM work.
  EnvWork = 0,
  //! @brief Cuda batching work that may spend significant time in scheduling GPU kernels or waiting for GPU
  //!        copies (HtoD and DtoH) to finish.
  BatchWork = 1
};

//! @brief Multi-threading start point: Queues up a batch of work defined by [start_index, end_index].
//! {@ref func} will be called with the provided {@ref arg} and each index in the range.
//! When the entire batch is done, {@ref batch_completion_cb} will be called if provided.
inline void add_work_batched(VecEnv* vec_env, const std::function<void(void*, int)>& func, void* arg, int start_index,
  int end_index, const std::function<void(void*)>& batch_completion_cb, int min_num_items_per_batch,
  PufferWorkType work_type)
{
#if defined(PUFFER_SINGLE_THREADED)
  for (int i = start_index; i <= end_index; i++) { func(arg, i); }
  if (batch_completion_cb != nullptr) { batch_completion_cb(arg); }
  return;
#endif
  auto* threading = work_type == PufferWorkType::EnvWork ? vec_env->threading_env : vec_env->threading_batch;

  PUFFER_ASSERT(threading != nullptr && end_index >= start_index && min_num_items_per_batch > 0,
    "Invalid state/params.");
  std::shared_ptr<BatchCompletion> batch_completion = {};
  if (batch_completion_cb != nullptr)
  {
    batch_completion = std::make_shared<BatchCompletion>(batch_completion_cb);
    batch_completion->batch_total_tasks.fetch_add(end_index - start_index + 1);
  }
  const auto num_threads = threading->num_threads.load();
  const auto num_work_items = end_index - start_index + 1;
  int batch_size = (num_work_items + num_threads) / num_threads;
  batch_size = std::min(num_work_items, std::max(min_num_items_per_batch, batch_size));
  threading->add_work({
    .func = func,
    .arg = arg,
    .start_index = start_index,
    .end_index = end_index,
    .batch_size = batch_size,
    .batch_completion = batch_completion
  });
}

// Overload without batch group for the C external API (not used by this cpp file).
void c_add_work_batched(VecEnv* vec_env, work_func func, void* arg, int start_index, int end_index)
{
  // `func` gets converted to std::function automatically a la `[func](args) { func(args); }`
  add_work_batched(vec_env, func, arg, start_index, end_index, nullptr, /* min_num_items_per_batch */ 16,
    PufferWorkType::EnvWork);
}

void c_wait_all_done(VecEnv* vec_env)
{
  PUFFER_ASSERT(vec_env->threading_env != nullptr && vec_env->threading_batch != nullptr, "Invalid threading state.");
  vec_env->threading_env->wait_all_done();
  vec_env->threading_batch->wait_all_done();
}
