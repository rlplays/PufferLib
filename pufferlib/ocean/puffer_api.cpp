#ifdef PUFFER_NATIVECPP_PYBINDINGS
#include <pybind11/gil.h>
#endif

struct LSTMWrapper;

struct PufferTorch
{
  // TODO(perumaal): Rename to LSTMEvalWrapper...
  LSTMWrapper* model;
  LSTMTrainWrapper* train_model;
};

void c_setup_pufferoptions(VecEnv* vec_env, const int num_actions, const int num_logits, const int input_size,
  const int hidden_size, const bool is_continuous, const int num_gpu_batches)
{
  PufferOptions* options = &vec_env->opts;
  options->num_actions = num_actions;
  options->num_logits = num_logits;
  options->logit_sizes = new int64_t[num_actions];
  options->num_gpu_batches = num_gpu_batches;
  options->num_threads_batch = std::min(options->num_threads_env, std::max(1, num_gpu_batches));

  for (int i = 0; i < num_actions; i++)
  {
    options->logit_sizes[i] = num_logits;
  }
  options->input_size = input_size;
  options->hidden_size = hidden_size;
  options->is_continuous = is_continuous;
}

void c_cleanup_pufferoptions(VecEnv* vec_env)
{
  if (vec_env->opts.logit_sizes)
  {
    DELETE_ARRAY(vec_env->opts.logit_sizes);
  }
  vec_env->opts = {};
}


PufferTorch* c_torch_alloc(VecEnv* vec_env)
{
  BEGIN_LIBTORCH_CATCH
  {
    PufferOptions* opts = &vec_env->opts;
    PUFFER_ASSERT(opts != nullptr && opts->num_actions > 0 && opts->num_atns == 0 && opts->logit_sizes != nullptr &&
      opts->enable_native_libtorch,
      "Invalid options.");
    auto* ptorch = new PufferTorch();
    ptorch->model = new LSTMWrapper(vec_env, opts, vec_env->num_envs);
    vec_env->puff_torch = ptorch;
    return ptorch;
  }
  END_LIBTORCH_CATCH
}

void c_torch_free(PufferTorch* pt)
{
  BEGIN_LIBTORCH_CATCH
  {
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr, "Invalid state.");
    DELETE_PTR(pt->model);
    DELETE_PTR(pt->train_model);
    delete pt;
  }
  END_LIBTORCH_CATCH
}

void c_torch_start_eval_lstm(uintptr_t vec_env_ptr, Tensor full_obs_cpu, Tensor full_rewards_cpu,
  Tensor full_terminals_cpu, Tensor encoder_linear_w, Tensor encoder_linear_b,
  Tensor decoder_linear_w, Tensor decoder_linear_b, Tensor value_w, Tensor value_b,
  Tensor weight_ih, Tensor weight_hh, Tensor bias_ih, Tensor bias_hh, Tensor obs_out,
  Tensor actions_out, Tensor logprobs_out, Tensor rewards_out, Tensor terminals_out,
  Tensor values_out)
{
  VecEnv* vec_env = (VecEnv*)vec_env_ptr;
  PufferTorch* puff_torch = vec_env->puff_torch;
  PUFFER_ASSERT(puff_torch != nullptr && puff_torch->model != nullptr, "Invalid state.");

  puff_torch->model->start_batch_eval_lstm(vec_env, full_obs_cpu, full_rewards_cpu, full_terminals_cpu,
    encoder_linear_w, encoder_linear_b, decoder_linear_w, decoder_linear_b,
    value_w, value_b, weight_ih, weight_hh, bias_ih, bias_hh, obs_out,
    actions_out, logprobs_out, rewards_out, terminals_out, values_out);
}

//! @brief Performs action (inference) + step segmented across a BPTT horizon batched by envs.
//! Waits for the entire horizon to finish. 
void c_torch_run_fulleval(uintptr_t vec_env_ptr)
{
  BEGIN_LIBTORCH_CATCH
  {
    RECORD_FUNCTION("torch_run_fulleval_cpp", std::vector<c10::IValue>({}));

    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferTorch* pt = vec_env->puff_torch;
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr && vec_env->num_envs > 0 && vec_env->envs != nullptr &&
      vec_env->threading_env != nullptr && vec_env->threading_batch != nullptr,
      "Invalid state/inputs.");
    pt->model->forward_eval_batch(vec_env);
  }
  END_LIBTORCH_CATCH
}

PufferEvalResult c_torch_finish_eval_lstm(uintptr_t vec_env_ptr)
{
  BEGIN_LIBTORCH_CATCH
  {
    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferTorch* pt = vec_env->puff_torch;
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr, "Invalid state.");
    return pt->model->finish_batch_eval_lstm(vec_env);
  }
  END_LIBTORCH_CATCH
}

void c_init_torch_train_lstm(uintptr_t vec_env_ptr, const PufferTrainOpts& train_opts)
{
  BEGIN_LIBTORCH_CATCH
  {
    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferOptions* opts = &vec_env->opts;
    PufferTorch* pt = vec_env->puff_torch;

    PUFFER_ASSERT(pt != nullptr, "Invalid state (PufferTorch is null).");
    // Some envs may have a driver env first which is not needed for training.
    if (pt->train_model != nullptr) { delete pt->train_model; }
    pt->train_model = new LSTMTrainWrapper(vec_env, opts, train_opts, vec_env->num_envs);
  }
  END_LIBTORCH_CATCH
}

PufferTrainResult c_torch_train_lstm(uintptr_t vec_env_ptr,
  int epoch, int total_epochs, int segments, int total_minibatches, int minibatch_segments, int accumulate_minibatches,
  Tensor obs, Tensor actions, Tensor logprobs, Tensor rewards, Tensor terminals, Tensor values)
{
#ifdef PUFFER_NATIVECPP_PYBINDINGS
  pybind11::gil_scoped_release no_gil; 
#endif
  BEGIN_LIBTORCH_CATCH
  {
    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferTorch* pt = vec_env->puff_torch;
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr, "Invalid state.");
    return pt->train_model->train_model(epoch, total_epochs, segments, total_minibatches, minibatch_segments, accumulate_minibatches,
      obs, actions, logprobs, rewards, terminals, values);
  }
  END_LIBTORCH_CATCH
}

LSTMTrainWrapper* get_train_wrapper(PufferTorch* pt)
{
  PUFFER_ASSERT(pt != nullptr, "PufferTorch is null.");
  if (pt->train_model == nullptr)
  {
    return nullptr;
  }
  return static_cast<LSTMTrainWrapper*>(pt->train_model);
}

PufferTrainWeights c_torch_train_get_weights(uintptr_t vec_env_ptr)
{
  BEGIN_LIBTORCH_CATCH
  {
    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferTorch* pt = vec_env->puff_torch;
    PUFFER_ASSERT(pt != nullptr && pt->train_model != nullptr, "Invalid state or train model not initialized.");
    return pt->train_model->get_weights();
  }
  END_LIBTORCH_CATCH
}

// Include the pybind layer if needed. Tests and other units can use this file without pulling in Pythin/pybind stuff.
#ifdef PUFFER_NATIVECPP_PYBINDINGS
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// Suggested by Claude to avoid pybind/C++ using import_array/numpy here while env_binding uses just the PyAPI alone
// (using non pybind).
#define PY_ARRAY_UNIQUE_SYMBOL puffer_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Forward declaration for env_glue.h stuff to avoid circular references. Especially as binding.c (C only)
// includes C code that wraps C++ code/objects underneath.
extern "C" PyMethodDef* get_c_env_binding_methods();

static inline void c_test_sample_logits(Tensor logits, int num_actions, std::vector<int64_t> logit_sizes,
  Tensor actions_out, Tensor logprobs_out)
{
  sample_logits(logits, num_actions, &logit_sizes[0], actions_out, logprobs_out);
}

static inline void c_test_sample_logits_entropy(Tensor logits, int num_actions, std::vector<int64_t> logit_sizes,
  Tensor actions_in, Tensor logprobs_out, Tensor entropy_out)
{
  sample_logits_entropy(logits, num_actions, &logit_sizes[0], actions_in, logprobs_out, entropy_out);
}

// Minimal version to test and match the Python <-> C++ versions.
// Runs on the main thread (as it's per batch).
static inline void c_single_batch_forward_pass(uintptr_t vec_env_ptr, int batch_index)
{
  auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
  auto* pt = vec_env->puff_torch;
  PUFFER_ASSERT(pt != nullptr && pt->model != nullptr, "Invalid state.");
  pt->model->copy_obs_forward_eval_batch(batch_index);
  pt->model->sync_cuda_stream(batch_index);
}

PYBIND11_MODULE(binding, m)
{
  m.doc() = "PufferLib Libtorch API";

  py::class_<PufferPerfStat>(m, "PufferPerfStat")
      .def(py::init<>())
      .def_readwrite("name", &PufferPerfStat::name)
      .def_readwrite("num_batches", &PufferPerfStat::num_batches)
      .def_readwrite("total_duration_ms", &PufferPerfStat::total_duration_ms)
      .def_readwrite("avg_us", &PufferPerfStat::avg_us)
      .def_readwrite("std_dev_us", &PufferPerfStat::std_dev_us)
      .def_readwrite("sample_us", &PufferPerfStat::sample_us);
  py::class_<PufferEvalResult>(m, "PufferEvalResult")
      .def(py::init<>())
      .def_readwrite("perf_stats", &PufferEvalResult::perf_stats)
      .def_readwrite("step_count", &PufferEvalResult::step_count)
      .def_readwrite("total_steps", &PufferEvalResult::total_steps);
  py::class_<PufferTrainOpts>(m, "PufferTrainOpts")
      .def(py::init<>())
      .def_readwrite("config", &PufferTrainOpts::config);

  py::class_<PufferTrainStat>(m, "PufferTrainStat")
      .def(py::init<>())
      .def_readwrite("name", &PufferTrainStat::name)
      .def_readwrite("value_str", &PufferTrainStat::value_str)
      .def_readwrite("value_int", &PufferTrainStat::value_int)
      .def_readwrite("value_bool", &PufferTrainStat::value_bool)
      .def_readwrite("value_dbl", &PufferTrainStat::value_dbl);

  py::class_<PufferTrainResult>(m, "PufferTrainResult")
      .def(py::init<>())
      .def_readwrite("perf_stats", &PufferTrainResult::perf_stats)
      .def_readwrite("train_stats", &PufferTrainResult::train_stats);

  py::class_<PufferTrainWeights>(m, "PufferTrainWeights")
      .def(py::init<>())
      .def_readwrite("encoder_w", &PufferTrainWeights::encoder_w)
      .def_readwrite("encoder_b", &PufferTrainWeights::encoder_b)
      .def_readwrite("decoder_w", &PufferTrainWeights::decoder_w)
      .def_readwrite("decoder_b", &PufferTrainWeights::decoder_b)
      .def_readwrite("value_w", &PufferTrainWeights::value_w)
      .def_readwrite("value_b", &PufferTrainWeights::value_b)
      .def_readwrite("lstm_weight_ih", &PufferTrainWeights::lstm_weight_ih)
      .def_readwrite("lstm_weight_hh", &PufferTrainWeights::lstm_weight_hh)
      .def_readwrite("lstm_bias_ih", &PufferTrainWeights::lstm_bias_ih)
      .def_readwrite("lstm_bias_hh", &PufferTrainWeights::lstm_bias_hh);

  import_array();
  PyModule_AddFunctions(m.ptr(), get_c_env_binding_methods());
  m.def("libtorch_info", &c_libtorch_info, "Print libtorch info to stdout.");
  m.def("sample_logits", &c_test_sample_logits, py::arg("logits"), py::arg("num_actions"), py::arg("logit_sizes"),
    py::arg("actions_out"), py::arg("logprobs_out"), "Test sample logits.");
  m.def("sample_logits_with_entropy", &c_test_sample_logits_entropy, py::arg("logits"), py::arg("num_actions"), py::arg("logit_sizes"),
    py::arg("actions_in"), py::arg("logprobs_out"), py::arg("entropy_out"), 
    "Test sample logits that outputs entropy/logprobs for given actions/logits.");

  m.def("torch_start_eval_lstm", &c_torch_start_eval_lstm, py::arg("vec_env"), py::arg("full_obs_cpu"),
    // Full observation tensor on CPU across all horizons/envs with shape [envs, horizon, obs_count].
    py::arg("full_rewards_cpu"),   // Full rewards tensor on CPU across all horizons/envs [envs, horizon, 1].
    py::arg("full_terminals_cpu"), // Full terminals tensor on CPU across all horizons/envs [envs, horizon, 1].
    py::arg("encoder_linear_w"), py::arg("encoder_linear_b"), py::arg("decoder_linear_w"),
    py::arg("decoder_linear_b"), py::arg("value_w"), py::arg("value_b"), py::arg("weight_ih"), py::arg("weight_hh"),
    py::arg("bias_ih"), py::arg("bias_hh"), py::arg("observations_out"), py::arg("actions_out"),
    py::arg("logprobs_out"), py::arg("rewards_out"), py::arg("terminals_out"), py::arg("values_out"),
    "Start the initial torch eval (before starting the horizon segments).");

  m.def("torch_run_fulleval", &c_torch_run_fulleval, py::arg("vec_env"),
    "Runs the full forward eval pass using libtorch for all segments in the horizon.");

  m.def("torch_run_single_eval", &c_single_batch_forward_pass, py::arg("vec_env"), py::arg("batch_index"),
    "Runs the full forward eval pass using libtorch for all segments in the horizon.");

  m.def("torch_finish_eval_lstm", &c_torch_finish_eval_lstm, py::arg("vec_env"),
    "Finish the torch eval (after all segments in the horizon are done).");

  m.def("torch_init_train_lstm", &c_init_torch_train_lstm, py::arg("vec_env"), py::arg("train_opts"),
    "Initialize the native LSTM training.");

  m.def("torch_train_lstm", &c_torch_train_lstm, py::arg("vec_env"), py::arg("epoch"),
    py::arg("total_epochs"), py::arg("segments"), py::arg("total_minibatches"), py::arg("minibatch_segments"),
    py::arg("accumulate_minibatches"), py::arg("obs"), py::arg("actions"), py::arg("logprobs"), py::arg("rewards"),
    py::arg("terminals"), py::arg("values"),
    "Train the LSTM model using the provided horizon trajectories.");

  m.def("torch_train_get_weights", &c_torch_train_get_weights, py::arg("vec_env"),
    "Get the current training weights from the LSTM model.");
}

#endif
