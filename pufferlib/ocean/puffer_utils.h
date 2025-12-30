// Utils used by puffer_native.cpp.
#include "puffer_native.h"

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

#if DEBUG
constexpr bool global_debug_mode = true;
#else
constexpr bool global_debug_mode = false;
#endif

#ifdef PUFFER_CUDA_MEMCHECK
void print_cuda_mem_info(std::string name, bool print_detailed = false,
  std::vector<std::tuple<std::string, Tensor>> tensors_to_check = {});
#else
// Completely eliminate any std::string ops etc for non-mem-check builds.
#define print_cuda_mem_info(_1, ...) ((void)0)
#endif


// LibTorch throws exceptions on errors, log them correctly in debug mode only.
#if DEBUG
#define BEGIN_LIBTORCH_CATCH try
#else
#define BEGIN_LIBTORCH_CATCH
#endif

#if DEBUG
#define END_LIBTORCH_CATCH                                                                                             \
  catch (const c10::Error& e)                                                                                          \
  {                                                                                                                    \
    std::cerr << "Error from libtorch: " << e.what() << std::endl;                                                     \
    PUFFER_ASSERT_BREAK();                                                                                             \
    throw;                                                                                                             \
  }

#else
#define END_LIBTORCH_CATCH
#endif


//
// LibTorch core functions.
//

void c_libtorch_info()
{
  std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
  std::cout << "cuDNN available: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;
  if (torch::cuda::is_available())
  {
    std::cout << "Number of CUDA devices: " << torch::cuda::device_count() << std::endl;
  }
  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  Tensor test_tensor = torch::zeros({2, 2}, device);
  std::cout << "Test tensor device: " << test_tensor.device() << std::endl;
}


// TOOD(perumaal): Move all these helpers out to unclunkyfy this file.
// Callable from Python to ensure Python<->C++ views are consistent and that no copies are needed.
void c_print_tensor_info(Tensor tensor, string name = "", bool print_values = false)
{
#if DEBUG
  const auto numel = tensor.numel();
  const auto elem_size = tensor.element_size();
  const double total_bytes = double(static_cast<std::uint64_t>(numel) * static_cast<std::uint64_t>(elem_size));
  const double total_mb = total_bytes / (1024.0 * 1024.0);

  std::ostringstream device_ss;
  device_ss << tensor.device();
  std::ostringstream dtype_ss;
  dtype_ss << tensor.dtype();
  std::ostringstream sizes_ss;
  sizes_ss << tensor.sizes();

  std::printf(
    "Tensor: %s  %s / %s / %s / %.3f MB ] [ptr 0x%p]\n", name.c_str(), device_ss.str().c_str(), dtype_ss.str().c_str(),
    sizes_ss.str().c_str(), total_mb, tensor.const_data_ptr());

  if (print_values)
  {
    // VERY Expensive to do this, so strictly for debugging.
    auto t = tensor.cpu();
    if (t.dim() >= 2)
    {
      const int64_t max0 = std::min<int64_t>(4, t.size(0));
      const int64_t max1 = std::min<int64_t>(8, t.size(1));

      t = t.narrow(0, 0, max0).narrow(1, 0, max1);
    }

    std::cout << name << " (showing only a small slice):\n{" << t << "}\n\n";
  }
#endif
}


void c_print_tensor_infos(Tensor tensor1, Tensor tensor2, string name)
{
  c_print_tensor_info(tensor1, "Tensor 1: " + name);
  c_print_tensor_info(tensor2, "Tensor 2: " + name);
}

void c_compare_tensors(Tensor tensor1, string name1, Tensor tensor2, string name2, float eps = 0.001f)
{
  c_print_tensor_info(tensor1, "Tensor 1: " + name1);
  c_print_tensor_info(tensor2, "Tensor 2: " + name2);
  if (tensor1.sizes() != tensor2.sizes())
  {
    std::cout << "Tensor shape mismatch for " << ": " << name1 << " " << tensor1.sizes() << " vs " << name2 << " " <<
        tensor2.sizes() << std::endl;
    return;
  }
  auto t1 = tensor1.cpu().flatten();
  auto t2 = tensor2.cpu().flatten();
  auto t1arr = t1.data_ptr<float>();
  auto t2arr = t2.data_ptr<float>();
  int j = 0;
  for (int i = 0; i < t1.numel(); i++)
  {
    const float v1 = t1arr[i];
    const float v2 = t2arr[i];
    const float diff = std::abs(v1 - v2);
    if (diff > eps)
    {
      std::cout << "Tensor mismatch " << name1 << ": #" << i << ": " << v1 << " vs " << v2
          << " (diff: " << diff << ")\n";
      if (++j >= 100) { return; }
    }
  }
#if defined(DEBUG)
  if (j == 0)
  {
    std::cout << "Tensors match for " << name1 << " / " << name2 << std::endl;
  }
#endif
}


// Whether to reserve lap times for performance calculations.

// Simple performance timer (NOT thread-safe, must ensure it's per-thread or per-batch).
struct PerfTimer
{
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  std::chrono::duration<double, std::nano> duration;
  std::string name;

  // Used to calculate stddev etc (optional, if lap_durations is sized > 1).
  std::vector<double> lap_durations;
  int ring_index = 0;
  int ring_count = 0;

  inline PerfTimer& start()
  {
    start_time = std::chrono::high_resolution_clock::now();
    return *this;
  }

  inline PerfTimer& stop()
  {
    end_time = std::chrono::high_resolution_clock::now();
    const auto dur = (end_time - start_time);
    if (lap_durations.size() > 1)
    {
      lap_durations[ring_index] = std::chrono::duration<double, std::nano>(dur).count();
      ring_index = (ring_index + 1) % lap_durations.size();
      ring_count++;
    }
    duration += dur;
    return *this;
  }

  inline PerfTimer& lap()
  {
    stop();
    start();
    return *this;
  }

  //! @brief (Slow) Calculates average and stddev of the lap durations (only if the laps ring buffer is filled).
  std::tuple<double, double> calc_avg_stddev_ns() const
  {
    if (lap_durations.empty()) return {0, 0};
    double sum_sq_ns = 0.0;
    // If N calls take M ns, it doesn't mean we will accurately get M/N for each call (as a function may perform sub-nanos ops), 
    // so we use the overall average calculated from total duration.
    size_t n = std::min(ring_count, (int)lap_durations.size());
    for (size_t i = 0; i < n; i++)
    {
      const auto v = lap_durations[i];
      sum_sq_ns += (v * v);
    }

    double mean_ns = duration.count() / n;
    double variance = (sum_sq_ns / (n - 1)) - (mean_ns * mean_ns); // sample stddev
    return {mean_ns, std::sqrt(variance)};
  }

  //! @brief (Slow) Formats microseconds into us/ms/s string.
  static std::string format_ns(const double ns)
  {
    if (ns > (1000.0 * 1000.0 * 1000.0)) { return std::to_string(ns / (1000.0 * 1000.0 * 1000.0)) + "s"; }
    if (ns > (1000.0 * 1000.0))
    {
      return std::to_string(ns / (1000.0 * 1000.0)) + "ms";
    }
    if (ns > 1000.0) { return std::to_string(ns / 1000.0) + "us"; }
    return std::to_string(ns) + "ns";
  }

  //! @brief (Slow) Prints the timer result in us/ms/s to stdout including optional iteration count.
  void print(const int iters = 1) const
  {
    auto n = name;
    if (n.size() > 16) { n = n.substr(0, 16); }
    std::cout << n << "\t took " << format_ns(duration.count());
    if (iters > 1 && lap_durations.size() > 1)
    {
      auto [avg_ns, stddev_ns] = calc_avg_stddev_ns();
      std::cout << "\t [ For " << iters << " iters; avg : " << format_ns(avg_ns) << "; stddev : " <<
          format_ns(stddev_ns) << " ]";
    }
    std::cout << "\n";
  }

  double get_duration_millis() const { return duration.count() / 1'000'000.0; }
};

//! @brief Starts and returns a PerfTimer with the given name.
/**
 Use it like this:
auto t1 = start_timer("name");
constexpr int iters = 10000;
for (int i = 0; i < COUNT; i++) {
  // ... code to time ...
}
t1.stop().print(COUNT);
*/
static PerfTimer make_timer(const std::string& name) { return PerfTimer{.name = name}; }

static PerfTimer start_timer_laps(const std::string& name, const int laps)
{
  return PerfTimer{.name = name, .lap_durations = std::vector<double>(laps)}.start();
}


struct PufferEvalResult
{
  // Perf stats (in ms) across all batches for this run.
  std::vector<std::tuple<std::string, double>> stats_millis;
  int64_t step_count;
  int64_t total_steps;
};


//! @brief Accumulates the given timer duration from different threads/batches into the result stats. 
//! Populates "name" with the average (divided by {@ref div_by}) and "name_sum" with the raw total sum
static void calc_total_perf_duration(int index, PufferEvalResult& result, PerfTimer& timer, double div_by)
{
  // Convert ns -> us.
  const double duration_us = (timer.duration.count() / 1000.0);
  auto name = timer.name + "_sum";
  double total_duration = -1;
  for (auto& stat : result.stats_millis)
  {
    if (std::get<0>(stat) == name)
    {
      std::get<1>(stat) += (duration_us / 1000.0);
      total_duration = std::get<1>(stat);
      break;
    }
  }
  result.stats_millis.push_back({timer.name + "_" + std::to_string(index), (duration_us / 1000.0)});
  if (total_duration < 0)
  {
    result.stats_millis.push_back({name, (total_duration = (duration_us / 1000.0))});
  }
  // Now calculate average.
  name = timer.name;
  for (auto& stat : result.stats_millis)
  {
    if (std::get<0>(stat) == name)
    {
      std::get<1>(stat) = (total_duration / div_by);
      return;
    }
  }
  result.stats_millis.push_back({name, (total_duration / div_by)});
}


static inline Tensor log_prob(Tensor logits, Tensor value)
{
  value = value.to(torch::kLong).unsqueeze(-1);
  auto res = torch::broadcast_tensors({value, logits});
  value = res[0];
  value = value.index({at::indexing::Ellipsis, at::indexing::Slice(0, 1)});
  auto log_pmf = res[1];
  log_pmf = log_pmf.gather(-1, value).squeeze(-1);
  res[0] = Tensor{};
  res[1] = Tensor{};
  return log_pmf;
}

//! @brief Returns a tuple of (actions, logprobs, entropy) sampled from the given raw logits.
//! Matches the Python version with optional entropy calculation (entropy might not be needed during eval for instance).
//! TODO(perumaal): Calc entropy and accept input actions during training.
static inline void sample_logits(Tensor logits, int num_actions, int64_t* logit_sizes,
  Tensor& actions, Tensor& logprobs)
{
  PUFFER_ASSERT(logits.dim() == 2, "Logits must be 2D (batch_size, total_num_logits).");

  actions.zero_();
  logprobs.zero_();
}


static void TestGPUBandwidth()
{
  BEGIN_LIBTORCH_CATCH

  {
    const auto mbs = {1, 2, 4, 8, 16, 32, 64};
    for (const auto& MB : mbs)
    {
      constexpr int COUNT = 100;
      auto t1 = start_timer_laps("gpu_transfer_" + std::to_string(MB) + "MB", COUNT);
      int tensor_size = (MB * 1024 * 1024) / sizeof(float);
      auto tensor = torch::zeros({tensor_size},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
      for (int i = 0; i < COUNT; i++)
      {
        auto t2 = tensor.to(torch::kCUDA);
        t1.lap();
      }
      t1.stop().print(COUNT);
      std::cout << "GB/s: " << ((double)MB/(t1.get_duration_millis() / double(COUNT))) << std::endl;
    }
    for (const auto& MB : mbs)
    {
      constexpr int COUNT = 100;
      auto t1 = start_timer_laps("gpu_transfer_pin_" + std::to_string(MB) + "MB", COUNT);
      int tensor_size = (MB * 1024 * 1024) / sizeof(float);
      auto tensor = torch::zeros({tensor_size},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).
          pin_memory();
      for (int i = 0; i < COUNT; i++)
      {
        auto t2 = tensor.to(torch::kCUDA);
        t1.lap();
      }
      t1.stop().print(COUNT);
      std::cout << "GB/s: " << ((double)MB/(t1.get_duration_millis() / double(COUNT))) << std::endl;
    }
  }
  END_LIBTORCH_CATCH
}

// Utility functions
#ifdef PUFFER_CUDA_MEMCHECK
static atomic_int num_cuda_mem_checks = 0;
constexpr int max_num_cuda_mem_checks = 256;

void print_cuda_mem_info(std::string name, bool print_detailed,
  std::vector<std::tuple<std::string, Tensor>> tensors_to_check)
{
  if (!torch::cuda::is_available()) return;
  num_cuda_mem_checks.fetch_add(1);
  if (num_cuda_mem_checks.load() > max_num_cuda_mem_checks) { return; }

  // Get memory info
  const c10::CachingDeviceAllocator::DeviceStats stats = CUDACachingAllocator::getDeviceStats(
    c10::cuda::current_device());

  auto alloc_bytes = 0.0;
  auto reserved_bytes = 0.0;
  auto active_allocs = 0;
  for (int i = 0; i < stats.allocated_bytes.size(); ++i)
  {
    alloc_bytes += stats.allocated_bytes[i].current;
    reserved_bytes += stats.reserved_bytes[i].current;
    active_allocs += stats.allocation[i].current;
  }
  std::cout << "Cuda mem stats: " << name << ":\t\t\t"
      << " [Allocated : " << (alloc_bytes / (1024.0 * 1024.0)) << " MB ]"
      << " [Reserved bytes: " << (reserved_bytes / (1024.0 * 1024.0)) << " MB ]"
      << " [Active allocs: " << active_allocs << "]\n";
  if (print_detailed)
  {
    size_t largestBlock = 0;
    CUDACachingAllocator::cacheInfo(c10::cuda::current_device(), &largestBlock);
    std::cout << "Cuda mem stats: " << name << "_detailed:\t"
        << " [Largest free block: " << (largestBlock / (1024.0 * 1024.0)) << " MB ]\n";
    // Get and print snapshot
    try
    {
      auto snapshot = CUDACachingAllocator::snapshot();

      std::cout << "Memory Snapshot for " << name << ":\n";
      std::cout << "  Device traces: " << snapshot.device_traces.size() << "\n";
      std::cout << "  Segments: " << snapshot.segments.size() << "\n";

      // Print top memory consuming segments
      size_t total_allocated = 0;
      size_t total_reserved = 0;
      int segment_count = 0;

      for (auto& seg : snapshot.segments)
      {
        total_allocated += seg.allocated_size;
        total_reserved += seg.total_size;
        if (seg.allocated_size > 1024 * 256)
        {
          std::cout << "    Segment " << segment_count
              << ": allocated=" << (seg.allocated_size / (1024.0 * 1024.0)) << " MB"
              << ", total=" << (seg.total_size / (1024.0 * 1024.0)) << " MB"
              << ", stream=" << seg.stream << "\n";
          // Try to associate tensors with this segment by pointer range.
          const auto seg_begin = uintptr_t(seg.address);
          const auto seg_end = seg_begin + seg.total_size;

          for (const auto& kv : tensors_to_check)
          {
            auto name = std::get<0>(kv);
            auto t = std::get<1>(kv);
            if (!t.defined()) { continue; }

            const auto* raw_ptr = t.data_ptr();
            if (raw_ptr == nullptr) { continue; }

            const auto tensor_addr = uintptr_t(raw_ptr);
            if (tensor_addr >= seg_begin && tensor_addr < seg_end)
            {
              c_print_tensor_info(t, "[Segment " + std::to_string(segment_count) + ": " + name + "]");
            }
          }
          ++segment_count;
        }
      }

      std::cout << "  Total allocated: " << (total_allocated / (1024.0 * 1024.0)) << " MB\n";
      std::cout << "  Total reserved: " << (total_reserved / (1024.0 * 1024.0)) << " MB\n";
    }
    catch (const std::exception& e)
    {
      std::cout << "Error getting snapshot: " << e.what() << "\n";
    }
  }
}
#endif
