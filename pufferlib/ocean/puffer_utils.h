#pragma once
// Utils used by puffer_native_eval/train.cpp.
#include "puffer_native_eval.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <string>

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

#if DEBUG
// Uncomment this to check CUDA fused kernels with their slower counterparts (evaluate both).
//#define PUFFER_DBG_CHECK_NETWORK_SLOW 1
//#define PUFFER_DBG_CHECK_COMPARE_BREAK 1
#endif

// To debug multi-threading issues, uncomment the following line to force single-threaded execution.
// Also helps when profiling memory via py/libtorch profiler as it shows only the main thread
// (the other threads are initialized way ahead of time).
//#define PUFFER_SINGLE_THREADED 1


// Uncomment this to print memory info while debugging.
//#define PUFFER_CUDA_MEMCHECK 1

#ifdef PUFFER_CUDA_MEMCHECK
void print_cuda_mem_info(std::string name, bool print_detailed = false,
  std::vector<std::tuple<std::string, Tensor>> tensors_to_check = {});
#else
// Completely eliminate any std::string ops etc for non-mem-check builds.
#define print_cuda_mem_info(_1, ...) ((void)0)

/**
Use as part of print_cuda_mem_info to show which values are being tracked.
{ {"h1", state->h1},  {"c1", state->c1},  {"h2", state->h2},  {"c2", state->c2},  {"logits", logits},
  {"values", values_out},  {"state->actions_cpu", state->actions_cpu},  {"state->rewards_cpu", state->rewards_cpu},
  {"state->terminals_cpu", state->terminals_cpu},  {"state->obs_device", obs_tensor},
  {"state->values_horizon_s", state->values_horizon[segment]},  {"final_obs", final_obs}  
}
*/

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
void print_tensor(Tensor tensor, string name = "", bool print_values = false)
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
  if (tensor.dtype() == torch::kFloat32 && tensor.numel() > 0) { dtype_ss << " [ Min: " << tensor.min().item<float>() << " Max: " << tensor.max().item<float>() << " Mean: " << tensor.mean().item<float>() << " Std: " << tensor.std().item<float>() << " ]"; }
  std::ostringstream sizes_ss;
  sizes_ss << tensor.sizes();
  std::ostringstream strides_ss;
  strides_ss << tensor.strides();

  auto sizes_str = sizes_ss.str();
  auto dtype_str = dtype_ss.str();
  auto device_str = device_ss.str();
  auto tensor_str = tensor.toString();
  auto strides_str = strides_ss.str();
  std::printf(
    "Tensor: %s  %s / dtype %s (%d bytes per elem) / [ %s / strides %s / %.3f MB ] [ptr 0x%p] (%s)\n", name.c_str(),
    device_str.c_str(), dtype_str.c_str(), (int)elem_size,
    sizes_str.c_str(), strides_str.c_str(), total_mb, tensor.const_data_ptr(), (tensor.requires_grad() ? "requires_grad" : "no_grad"));
  if (print_values)
  {
    // VERY Expensive to do this, so strictly for debugging.
    auto t = tensor.detach().cpu();
    if (t.dim() >= 2) { t = t.flatten(); }
    t = t.narrow(0, 0, std::min<int64_t>(50, t.size(0)));

    std::cout << std::fixed << std::setprecision(10);
    std::cout << name << " [";
    auto accessor = t.accessor<float, 1>();
    for (int64_t i = 0; i < t.size(0); i++) {
      if (i > 0) std::cout << ", ";
      std::cout << accessor[i];
    }
    std::cout << "]\n\n";
  }
#endif
}


void print_tensors(Tensor tensor1, Tensor tensor2, string name, bool print_values = false)
{
  print_tensor(tensor1, "Tensor 1: " + name, print_values);
  print_tensor(tensor2, "Tensor 2: " + name, print_values);
}

template <class T>
bool c_compare_tensors(Tensor tensor1, string name1, Tensor tensor2, string name2, bool print_values, T eps,
  bool break_on_mismatch)
{
  print_tensor(tensor1, "Tensor 1: " + name1, print_values);
  print_tensor(tensor2, "Tensor 2: " + name2, print_values);
  if (tensor1.sizes() != tensor2.sizes())
  {
    std::cout << "Tensor shape mismatch for " << ": " << name1 << " " << tensor1.sizes() << " vs " << name2 << " " <<
        tensor2.sizes() << std::endl;
    return false;
  }
  auto t1 = tensor1.cpu().flatten();
  auto t2 = tensor2.cpu().flatten();
  auto t1arr = static_cast<T*>(t1.data_ptr());
  auto t2arr = static_cast<T*>(t2.data_ptr());
  int j = 0;
  for (int i = 0; i < t1.numel(); i++)
  {
    const T v1 = t1arr[i];
    const T v2 = t2arr[i];
    const T diff = std::abs(v1 - v2);
    if (diff > eps)
    {
      std::cout << "Tensor mismatch " << name1 << ": #" << i << ": " << v1 << " vs " << v2
          << " (diff: " << diff << ")\n";
#if defined(PUFFER_DBG_CHECK_COMPARE_BREAK)
      PUFFER_ASSERT(!break_on_mismatch || j <= 5, "Breaking on tensor compare mismatches (hit 5 mismatches).");
#endif
      if (++j >= 100) { return false; }
    }
  }
#if defined(DEBUG)
  if (j == 0)
  {
    std::cout << "Tensors match for " << name1 << " / " << name2 << std::endl;
  }
#endif
  return j == 0;
}

template <class T>
void c_check_sentinel(Tensor tensor1, string name1, T sentinel_val)
{
  const auto t = tensor1.cpu().flatten();
  for (int i = 0; i < t.numel(); i++)
  {
    const T v1 = static_cast<T*>(t.data_ptr())[i];
    if (v1 == sentinel_val)
    {
      std::cout << "Tensor sentinel value found in " << name1 << " : #" << i << ": " << v1 << " != " << sentinel_val
          << "\n";
    }
  }
}

bool c_compare_tensorsf(Tensor tensor1, string name1, Tensor tensor2, string name2, bool print_values = false,
  float eps = 0.0001f, bool break_on_mismatch = true)
{
  return c_compare_tensors<float>(tensor1, name1, tensor2, name2, print_values, eps, break_on_mismatch);
}

bool c_compare_tensorsi(Tensor tensor1, string name1, Tensor tensor2, string name2, bool print_values = false,
  bool break_on_mismatch = true)
{
  return c_compare_tensors<long>(tensor1, name1, tensor2, name2, print_values, 0, break_on_mismatch);
}

// Whether to reserve lap times for performance calculations.

// Simple performance timer (NOT thread-safe, must ensure it's per-thread or per-batch).
struct PerfTimer
{
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  std::chrono::duration<double, std::nano> duration_ns;
  std::string name;

  // Used to calculate stddev etc (optional, if lap_durations is sized > 1).
  std::vector<double> lap_durations_ns;
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
    if (lap_durations_ns.size() > 1)
    {
      lap_durations_ns[ring_index] = std::chrono::duration<double, std::nano>(dur).count();
      ring_index = (ring_index + 1) % lap_durations_ns.size();
      ring_count++;
    }
    duration_ns += dur;
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
    if (lap_durations_ns.empty()) return {0, 0};
    double sum_sq_ns = 0.0;
    // If N calls take M ns, it doesn't mean we will accurately get M/N for each call (as a function may perform sub-nanos ops), 
    // so we use the overall average calculated from total duration.
    size_t n = std::min(ring_count, (int)lap_durations_ns.size());
    for (size_t i = 0; i < n; i++)
    {
      const auto v = lap_durations_ns[i];
      sum_sq_ns += (v * v);
    }

    // mean should use the expanded ring_count btw: The ring buffer size (n) might be smaller than ring_count.
    // mean is also a bit more 'precise' than the variance as we only collect a small sample not the whole population.
    double mean_ns = duration_ns.count() / (double)ring_count;
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
    if (n.size() > 32) { n = n.substr(0, 32); }
    else { n.append(32 - n.size(), ' '); }
    std::cout << n << "\t took " << format_ns(duration_ns.count());
    if (iters > 1 && lap_durations_ns.size() > 1)
    {
      auto [avg_ns, stddev_ns] = calc_avg_stddev_ns();
      std::cout << "\t [ For " << iters << " iters; avg : " << format_ns(avg_ns) << "; stddev : " <<
          format_ns(stddev_ns) << " ]";
    }
    std::cout << "\n";
  }

  double get_duration_millis() const { return duration_ns.count() / 1'000'000.0; }
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
static PerfTimer make_timer(const std::string& name, const int laps)
{
  return PerfTimer{.name = name, .lap_durations_ns = std::vector<double>(laps)};
}

[[maybe_unused]] static PerfTimer start_timer_laps(const std::string& name, const int laps) {
    return make_timer(name, laps).start();
}

#define MICROBENCH_START(name, count)           \
  {                                             \
    constexpr int COUNT = count;                \
    auto timer = start_timer_laps(#name, COUNT);\
    for (int i = 0; i < COUNT; i++)             \
    {

#define MICROBENCH_END()      \
      timer.lap();            \
    }                         \
    timer.stop().print(COUNT);\
  }


struct PufferPerfStat
{
  std::string name;
  // Use num_batches to calculate average duration per batch (as they may be overlapping).
  int num_batches;
  double total_duration_ms;
  // These are per-batch stats.
  std::vector<double> avg_us;
  std::vector<double> std_dev_us;
  // These are per-patch/per-segment samples.
  std::vector<double> sample_us;
};

struct PufferEvalResult
{
  // Perf stats (in ms) across all batches for this run.
  std::vector<PufferPerfStat> perf_stats;
  int64_t step_count;
  int64_t total_steps;
};

struct PufferTrainWeights
{
  Tensor encoder_w;
  Tensor encoder_b;
  Tensor decoder_w;
  Tensor decoder_b;
  Tensor value_w;
  Tensor value_b;
  Tensor lstm_weight_ih;
  Tensor lstm_weight_hh;
  Tensor lstm_bias_ih;
  Tensor lstm_bias_hh;
};

struct PufferTrainOpts
{
  std::map<std::string, std::string> config;

  // Accessors filled in by GPT 5.2
  bool has(const std::string& key) const { return config.find(key) != config.end(); }

  std::string get_str(const std::string& key, const std::string& def = "") const
  {
    const auto it = config.find(key);
    return (it == config.end()) ? def : it->second;
  }

  int get_int(const std::string& key, const int def = 0) const
  {
    const auto it = config.find(key);
    if (it == config.end()) return def;
    return parse_int_(it->second, def);
  }

  double get_double(const std::string& key, const double def = 0.0) const
  {
    const auto it = config.find(key);
    if (it == config.end()) return def;
    return parse_double_(it->second, def);
  }

  bool get_bool(const std::string& key, const bool def = false) const
  {
    const auto it = config.find(key);
    if (it == config.end()) return def;
    return parse_bool_(it->second, def);
  }

private:
  static inline std::string trim_(std::string s)
  {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
  }

  static inline std::string lower_(std::string s)
  {
    std::transform(s.begin(), s.end(), s.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
  }

  static inline int parse_int_(const std::string& raw, const int def)
  {
    std::string s = trim_(raw);
    if (s.empty()) return def;

    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') return def;
    return static_cast<int>(v);
  }

  static inline double parse_double_(const std::string& raw, const double def)
  {
    std::string s = trim_(raw);
    if (s.empty()) return def;

    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (errno != 0 || end == s.c_str() || *end != '\0') return def;
    return v;
  }

  static inline bool parse_bool_(const std::string& raw, const bool def)
  {
    const std::string s = lower_(trim_(raw));
    if (s.empty()) return def;

    if (s == "1" || s == "true" || s == "t" || s == "yes" || s == "y" || s == "on") return true;
    if (s == "0" || s == "false" || s == "f" || s == "no" || s == "n" || s == "off") return false;
    return def;
  }
};

struct PufferTrainStat
{
  std::string name;
  std::string value_str;
  double value_dbl;
  int value_int;
  bool value_bool;
};

struct PufferTrainResult
{
  // Perf stats (in ms) across all batches for this run.
  std::vector<PufferPerfStat> perf_stats;
  std::vector<PufferTrainStat> train_stats;
};

[[nodiscard]]
static torch::nn::Linear layer_init(torch::nn::Linear layer, const double std = std::sqrt(2.0),
  const double bias_const = 0.0)
{
  torch::nn::init::orthogonal_(layer->weight, std);
  torch::nn::init::constant_(layer->bias, bias_const);
  return layer;
}

static void assign_tensors(Tensor& to, const Tensor from, string name)
{

#if DEBUG
  // print_tensors(to, from, "to (1) <- from (2) " + name);
  PUFFER_ASSERT(from.sizes() == to.sizes(), "Tensor size mismatch.");
  PUFFER_ASSERT(from.dim() == to.dim(), "Tensor dims mismatch.");
#endif

  torch::NoGradGuard no_grad;
  Tensor src = from.detach();
  if (src.device() != to.device()) src = src.to(to.device());
  if (src.scalar_type() != to.scalar_type()) src = src.to(to.scalar_type());
  if (!src.is_contiguous()) src = src.contiguous();

  to.copy_(src);
}


//! @brief Accumulates the given timer duration from different threads/batches into the result stats. 
static void calc_total_perf_duration(int index, PufferEvalResult& result, PerfTimer& timer, int num_batches)
{
  const double duration_ms = (timer.duration_ns.count() / (1000.0 * 1000.0));
  auto name = timer.name;
  PufferPerfStat* stat_ptr = nullptr;
  // It's okay, it's just a few elements, do 2 linear searches instead of complicated maps and stuff.
  for (auto& stat : result.perf_stats)
  {
    if (stat.name == name)
    {
      stat_ptr = &stat;
      break;
    }
  }
  if (stat_ptr == nullptr)
  {
    result.perf_stats.push_back({name, 0, 0.0, {}, {}, {}});
    stat_ptr = &result.perf_stats.back();
  }
  stat_ptr->total_duration_ms += duration_ms;
  stat_ptr->num_batches = num_batches;
  for (const auto s : timer.lap_durations_ns) { stat_ptr->sample_us.push_back(s / 1000.0); }
  auto [avg_ns, std_dev_ns] = timer.calc_avg_stddev_ns();
  stat_ptr->avg_us.push_back(avg_ns / 1000.0);
  stat_ptr->std_dev_us.push_back(std_dev_ns / 1000.0);
}


#if DEBUG
static void DBG_CHECK_LOGITS_INPUT(Tensor logits, int num_actions, int64_t* logit_sizes,
  Tensor actions_out, Tensor logprobs_out)
{
  int total_logit_size = 0;
  for (int i = 0; i < num_actions; i++) { total_logit_size += logit_sizes[i]; }
  PUFFER_ASSERT(logits.dim() == 2, "Logits must be 2D (batch_size, total_num_logits) discrete.");
  if (num_actions == 1)
  {
    PUFFER_ASSERT(actions_out.sizes() == at::IntArrayRef({logits.sizes()[0]}),
      "Actions (discrete) tensor size mismatch.");
  }
  else
  {
    PUFFER_ASSERT(actions_out.sizes() == at::IntArrayRef({logits.sizes()[0], num_actions}),
      "Actions (multidiscrete) tensor size mismatch.");
  }
  PUFFER_ASSERT(logprobs_out.sizes() == at::IntArrayRef{logits.sizes()[0]},
    "Logprobs tensor must match actions tensor size.");
}

static void DBG_CHECK_LOGITS_OUTPUT(Tensor logits, int num_actions, int64_t* logit_sizes,
  Tensor actions_out, Tensor logprobs_out)
{
  actions_out = actions_out.to(torch::kCPU).to(torch::kInt32);
  auto* actions = static_cast<int*>(actions_out.data_ptr());
  for (int64_t i = 0; i < actions_out.size(0); i++)
  {
    if (num_actions > 1) { PUFFER_ASSERT(actions_out.size(1) == num_actions, "Must match number of actions."); }
    for (int64_t j = 0; j < num_actions; j++)
    {
      const int action = (num_actions == 1) ? actions[i] : actions[i * num_actions + j];
      PUFFER_ASSERT(action >= 0, "Action must be >= 0.");
      PUFFER_ASSERT(action < logit_sizes[j], "Action must be < logit_sizes.");
    }
  }
}
#else
#define DBG_CHECK_LOGITS_INPUT(_1, _2, _3, _4, _5) ((void)0)
#define DBG_CHECK_LOGITS_OUTPUT(_1, _2, _3, _4, _5) ((void)0)
#endif

//! @brief Returns a tuple of (actions, logprobs, entropy) sampled from the given raw logits.
//! Matches the Python version with optional entropy calculation (entropy might not be needed during eval for instance).
//! TODO(perumaal): Calc entropy and accept input actions during training.
static void sample_logits(Tensor logits, int num_actions, int64_t* logit_sizes,
  Tensor actions_out, Tensor logprobs_out)
{
  DBG_CHECK_LOGITS_INPUT(logits, num_actions, logit_sizes, actions_out, logprobs_out);
  if (num_actions > 1)
  {
    logits = logits.reshape(at::IntArrayRef({logits.size(0), num_actions, static_cast<int>(logit_sizes[0])}));
  }
  logits = torch::nan_to_num(logits);
  auto logprobs = torch::log_softmax(logits, -1);
  auto probs = logprobs.exp();
  if (num_actions > 1)
  {
    probs = probs.reshape(at::IntArrayRef({-1, probs.size(-1)}));
  }
  auto action = at::multinomial(probs, 1, true).to(torch::kInt32);
  Tensor logprob;
  if (num_actions == 1)
  {
    logprob = logprobs.gather(-1, action).squeeze(-1);
    action = action.squeeze(1);
  }
  else
  {
    action = action.squeeze().reshape(at::IntArrayRef({logits.size(0), logits.size(1)}));
    logprob = logprobs.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    logprob = logprob.sum(-1);
  }
  PUFFER_ASSERT(action.dtype() == actions_out.dtype(), "Must match final actions' dtype.");
  actions_out.copy_(action);

  logprobs_out.copy_(logprob);
  DBG_CHECK_LOGITS_OUTPUT(logits, num_actions, logit_sizes, actions_out, logprobs_out);
}

//! @brief Uses the newly calculated logits and the existing actions based on the observations to
//! then calculate new logprobs/entropy (negative, so we can explore more I guess?).
static void sample_logits_entropy(Tensor logits, int num_actions, int64_t* logit_sizes,
  const Tensor actions, Tensor& logprobs_out, Tensor& entropy_out)
{
  if (num_actions > 1)
  {
    logits = logits.reshape(at::IntArrayRef({logits.size(0), num_actions, static_cast<int>(logit_sizes[0])}));
  }
  logits = torch::nan_to_num(logits);
  auto logprobs = torch::log_softmax(logits, -1);
  auto probs = logprobs.exp();
  Tensor p_log_p = -(probs * logprobs);
  p_log_p = p_log_p.sum(-1);
  int B = logits.size(0);
  Tensor actions_view = actions.view(at::IntArrayRef{B, -1});
  Tensor logprob;
  if (num_actions == 1)
  {
    p_log_p = p_log_p.squeeze(-1);
  }
  else { p_log_p = p_log_p.sum(-1); }
  
  entropy_out = p_log_p; 
  if (num_actions == 1)
  {
    logprob = logprobs.gather(-1, actions_view).squeeze(-1);
  }
  else
  {
    logprob = logprobs.gather(-1, actions_view.unsqueeze(-1)).squeeze(-1);
    logprob = logprob.sum(-1);
  }
  logprobs_out = logprob;
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
              print_tensor(t, "[Segment " + std::to_string(segment_count) + ": " + name + "]");
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
