// bindings.cpp - Python/Torch bindings for pufferlib
// Separated from pufferlib.cpp to allow clean inclusion for profiling

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include "pufferlib.cpp"

using namespace pufferlib;
namespace py = pybind11;

// Wrapper functions for Python bindings
pybind11::dict log_environments(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    Dict* out = log_environments_impl(pufferl);
    pybind11::dict py_out;
    for (int i = 0; i < out->size; i++) {
        py_out[out->items[i].key] = out->items[i].value;
    }
    return py_out;
}

Tensor initial_state(pybind11::object pufferl_obj, int64_t batch_size, torch::Device device) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    return pufferl.policy_bf16->initial_state(batch_size, device);
}

void python_vec_recv(pybind11::object pufferl_obj, int buf) {
    // Not used in static/OMP path
}

void python_vec_send(pybind11::object pufferl_obj, int buf) {
    // Not used in static/OMP path
}

torch::autograd::tensor_list env_buffers(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    return {pufferl.env.obs, pufferl.env.actions, pufferl.env.rewards, pufferl.env.terminals};
}

void rollouts(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::gil_scoped_release no_gil;
    auto t0 = std::chrono::high_resolution_clock::now();
    static_vec_omp_step(pufferl.vec);
    float sec = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t0).count();
    pufferl.profile.accum[PROF_ROLLOUT] += sec * 1000.0f;  // store as ms

    float eval_prof[NUM_EVAL_PROF];
    static_vec_read_profile(pufferl.vec, eval_prof);
    pufferl.profile.accum[PROF_EVAL_GPU] += eval_prof[EVAL_GPU];
    pufferl.profile.accum[PROF_EVAL_ENV] += eval_prof[EVAL_ENV_STEP];
}

pybind11::dict train(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    {
        pybind11::gil_scoped_release no_gil;
        train_impl(pufferl);
    }
    pybind11::dict losses;
    return losses;
}

pybind11::dict log_losses(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    Tensor losses_cpu = pufferl.losses.cpu();
    float* data = losses_cpu.data_ptr<float>();
    float n = data[LOSS_N];
    pybind11::dict result;
    if (n > 0) {
        float inv_n = 1.0f / n;
        result["pg_loss"] = data[LOSS_PG] * inv_n;
        result["vf_loss"] = data[LOSS_VF] * inv_n;
        result["entropy"] = data[LOSS_ENT] * inv_n;
        result["total_loss"] = data[LOSS_TOTAL] * inv_n;
        result["old_approx_kl"] = data[LOSS_OLD_APPROX_KL] * inv_n;
        result["approx_kl"] = data[LOSS_APPROX_KL] * inv_n;
        result["clipfrac"] = data[LOSS_CLIPFRAC] * inv_n;
    }
    pufferl.losses.zero_();
    return result;
}

pybind11::dict log_profile(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::dict result;
    float train_total = 0;
    for (int i = 0; i < NUM_PROF; i++) {
        float sec = pufferl.profile.accum[i] / 1000.0f;  // ms -> seconds
        result[PROF_NAMES[i]] = sec;
        if (i >= PROF_TRAIN_MISC) train_total += sec;
    }
    result["train"] = train_total;
    memset(pufferl.profile.accum, 0, sizeof(pufferl.profile.accum));
    return result;
}

pybind11::dict log_utilization(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::dict result;

    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(pufferl.nvml_device, &util);
    result["gpu_util"] = (float)util.gpu;

    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(pufferl.nvml_device, &mem);
    result["gpu_mem"] = 100.0f * (float)mem.used / (float)mem.total;

    size_t cuda_free, cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    result["vram_used_gb"] = (float)(cuda_total - cuda_free) / (1024.0f * 1024.0f * 1024.0f);
    result["vram_total_gb"] = (float)cuda_total / (1024.0f * 1024.0f * 1024.0f);

    // CPU memory from /proc/self/status
    long rss_kb = 0;
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) break;
        }
        fclose(f);
    }
    result["cpu_mem_gb"] = (float)rss_kb / (1024.0f * 1024.0f);

    return result;
}

void puf_close(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    close_impl(pufferl);
}

double get_config(py::dict& kwargs, const char* key) {
    if (!kwargs.contains(key)) {
        throw std::runtime_error(std::string("Missing config key: ") + key);
    }
    try {
        return kwargs[key].cast<double>();
    } catch (const py::cast_error& e) {
        throw std::runtime_error(std::string("Failed to cast config key '") + key + "': " + e.what());
    }
}

Dict* py_dict_to_c_dict(py::dict py_dict) {
    Dict* c_dict = create_dict(py_dict.size());
    for (auto item : py_dict) {
        const char* key = PyUnicode_AsUTF8(item.first.ptr());
        try {
            dict_set(c_dict, key, item.second.cast<double>());
        } catch (const py::cast_error&) {
            // Skip non-numeric values
        }
    }
    return c_dict;
}

std::unique_ptr<pufferlib::PuffeRL> create_pufferl(pybind11::dict kwargs, pybind11::dict vec_kwargs, pybind11::dict env_kwargs, pybind11::dict policy_kwargs) {
    HypersT hypers;
    // Layout (total_agents and num_buffers come from vec config)
    hypers.total_agents = get_config(vec_kwargs, "total_agents");
    hypers.num_buffers = get_config(vec_kwargs, "num_buffers");
    hypers.num_threads = get_config(vec_kwargs, "num_threads");
    hypers.horizon = get_config(kwargs, "horizon");
    // Model architecture (num_atns computed from env in C++)
    hypers.hidden_size = get_config(policy_kwargs, "hidden_size");
    hypers.num_layers = get_config(policy_kwargs, "num_layers");
    // Learning rate
    hypers.lr = get_config(kwargs, "learning_rate");
    hypers.min_lr_ratio = get_config(kwargs, "min_lr_ratio");
    hypers.anneal_lr = get_config(kwargs, "anneal_lr");
    // Optimizer
    hypers.beta1 = get_config(kwargs, "beta1");
    hypers.beta2 = get_config(kwargs, "beta2");
    hypers.eps = get_config(kwargs, "eps");
    // Training
    hypers.minibatch_size = get_config(kwargs, "minibatch_size");
    hypers.replay_ratio = get_config(kwargs, "replay_ratio");
    hypers.total_timesteps = get_config(kwargs, "total_timesteps");
    hypers.max_grad_norm = get_config(kwargs, "max_grad_norm");
    // PPO
    hypers.clip_coef = get_config(kwargs, "clip_coef");
    hypers.vf_clip_coef = get_config(kwargs, "vf_clip_coef");
    hypers.vf_coef = get_config(kwargs, "vf_coef");
    hypers.ent_coef = get_config(kwargs, "ent_coef");
    // GAE
    hypers.gamma = get_config(kwargs, "gamma");
    hypers.gae_lambda = get_config(kwargs, "gae_lambda");
    // VTrace
    hypers.vtrace_rho_clip = get_config(kwargs, "vtrace_rho_clip");
    hypers.vtrace_c_clip = get_config(kwargs, "vtrace_c_clip");
    // Priority
    hypers.prio_alpha = get_config(kwargs, "prio_alpha");
    hypers.prio_beta0 = get_config(kwargs, "prio_beta0");
    // Flags
    hypers.use_rnn = get_config(kwargs, "use_rnn");
    hypers.cudagraphs = get_config(kwargs, "cudagraphs");
    hypers.kernels = get_config(kwargs, "kernels");
    hypers.profile = get_config(kwargs, "profile");
    // Multi-GPU
    hypers.rank = get_config(kwargs, "rank");
    hypers.world_size = get_config(kwargs, "world_size");
    hypers.nccl_id_path = kwargs["nccl_id_path"].cast<std::string>();

    std::string env_name = kwargs["env_name"].cast<std::string>();
    Dict* vec_dict = py_dict_to_c_dict(vec_kwargs.cast<py::dict>());
    Dict* env_dict = py_dict_to_c_dict(env_kwargs.cast<py::dict>());

    pybind11::gil_scoped_release no_gil;
    return create_pufferl_impl(hypers, env_name, vec_dict, env_dict);
}

TORCH_LIBRARY(pufferlib, m) {
   m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, float gamma, float lambda, float rho_clip, float c_clip) -> ()");
}

TORCH_LIBRARY_IMPL(pufferlib, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cuda);
}

TORCH_LIBRARY(_C, m) {
    m.def("mingru_gate(Tensor state, Tensor combined) -> (Tensor, Tensor)");
    m.def("fc_max(Tensor x, Tensor W, Tensor b) -> Tensor");
}

PYBIND11_MODULE(_C, m) {
    m.def("log_environments", &log_environments);
    m.def("log_losses", &log_losses);
    m.def("log_profile", &log_profile);
    m.def("log_utilization", &log_utilization);
    m.def("rollouts", &rollouts);
    m.def("train", &train);
    m.def("close", &puf_close);
    m.def("logcumsumexp_cuda", &logcumsumexp_cuda);
    m.def("initial_state", &initial_state);
    m.def("mingru_gate", &mingru_gate);
    m.def("fc_max", &fc_max);
    m.def("fc_max_cpp", &fc_max_cpp);
    m.def("sample_logits", &sample_logits);
    m.def("python_vec_recv", &python_vec_recv);
    m.def("python_vec_send", &python_vec_send);
    m.def("env_buffers", &env_buffers);
    m.def("profiler_start", &profiler_start);
    m.def("profiler_stop", &profiler_stop);

    py::class_<Muon>(m, "Muon")
        .def_readwrite("lr", &Muon::lr)
        .def_readwrite("weight_buffer", &Muon::weight_buffer)
        .def_readwrite("momentum_buffer", &Muon::momentum_buffer)
        .def("state_dict", &Muon::state_dict)
        .def("load_state_dict", &Muon::load_state_dict);

    py::class_<HypersT>(m, "HypersT")
        .def_readwrite("horizon", &HypersT::horizon)
        .def_readwrite("total_agents", &HypersT::total_agents)
        .def_readwrite("num_buffers", &HypersT::num_buffers)
        .def_readwrite("num_atns", &HypersT::num_atns)
        .def_readwrite("hidden_size", &HypersT::hidden_size)

        .def_readwrite("replay_ratio", &HypersT::replay_ratio)
        .def_readwrite("num_layers", &HypersT::num_layers)
        .def_readwrite("lr", &HypersT::lr)
        .def_readwrite("min_lr_ratio", &HypersT::min_lr_ratio)
        .def_readwrite("anneal_lr", &HypersT::anneal_lr)
        .def_readwrite("beta1", &HypersT::beta1)
        .def_readwrite("beta2", &HypersT::beta2)
        .def_readwrite("eps", &HypersT::eps)
        .def_readwrite("total_timesteps", &HypersT::total_timesteps)
        .def_readwrite("max_grad_norm", &HypersT::max_grad_norm)
        .def_readwrite("clip_coef", &HypersT::clip_coef)
        .def_readwrite("vf_clip_coef", &HypersT::vf_clip_coef)
        .def_readwrite("vf_coef", &HypersT::vf_coef)
        .def_readwrite("ent_coef", &HypersT::ent_coef)
        .def_readwrite("gamma", &HypersT::gamma)
        .def_readwrite("gae_lambda", &HypersT::gae_lambda)
        .def_readwrite("vtrace_rho_clip", &HypersT::vtrace_rho_clip)
        .def_readwrite("vtrace_c_clip", &HypersT::vtrace_c_clip)
        .def_readwrite("prio_alpha", &HypersT::prio_alpha)
        .def_readwrite("prio_beta0", &HypersT::prio_beta0)
        .def_readwrite("use_rnn", &HypersT::use_rnn)
        .def_readwrite("cudagraphs", &HypersT::cudagraphs)
        .def_readwrite("kernels", &HypersT::kernels)
        .def_readwrite("profile", &HypersT::profile)
        .def_readwrite("rank", &HypersT::rank)
        .def_readwrite("world_size", &HypersT::world_size)
        .def_readwrite("nccl_id_path", &HypersT::nccl_id_path);

    py::class_<RolloutBuf>(m, "RolloutBuf")
        .def_readwrite("observations", &RolloutBuf::observations)
        .def_readwrite("actions", &RolloutBuf::actions)
        .def_readwrite("values", &RolloutBuf::values)
        .def_readwrite("logprobs", &RolloutBuf::logprobs)
        .def_readwrite("rewards", &RolloutBuf::rewards)
        .def_readwrite("terminals", &RolloutBuf::terminals)
        .def_readwrite("ratio", &RolloutBuf::ratio)
        .def_readwrite("importance", &RolloutBuf::importance);

    m.def("create_pufferl", &create_pufferl);
    py::class_<PuffeRL, std::unique_ptr<PuffeRL>>(m, "PuffeRL")
        .def_readwrite("policy_bf16", &PuffeRL::policy_bf16)
        .def_readwrite("policy_fp32", &PuffeRL::policy_fp32)
        .def_readwrite("muon", &PuffeRL::muon)
        .def_readwrite("hypers", &PuffeRL::hypers)
        .def_readwrite("rollouts", &PuffeRL::rollouts);

    py::class_<PolicyLSTM, std::shared_ptr<PolicyLSTM>, torch::nn::Module> cls(m, "PolicyLSTM");
    cls.def(py::init<int, int, int>());
    cls.def("forward", &PolicyLSTM::forward);
    cls.def("forward_train", &PolicyLSTM::forward_train);

    py::class_<Logits>(m, "Logits")
        .def_readwrite("mean", &Logits::mean)
        .def_readwrite("logstd", &Logits::logstd);

    py::class_<Encoder, std::shared_ptr<Encoder>, torch::nn::Module>(m, "Encoder");
    py::class_<Decoder, std::shared_ptr<Decoder>, torch::nn::Module>(m, "Decoder");
    py::class_<DefaultEncoder, std::shared_ptr<DefaultEncoder>, Encoder>(m, "DefaultEncoder")
        .def(py::init<int, int>());
    py::class_<SnakeEncoder, std::shared_ptr<SnakeEncoder>, Encoder>(m, "SnakeEncoder")
        .def(py::init<int, int, int>());
    py::class_<G2048Encoder, std::shared_ptr<G2048Encoder>, Encoder>(m, "G2048Encoder")
        .def(py::init<int, int>());
    py::class_<DefaultDecoder, std::shared_ptr<DefaultDecoder>, Decoder>(m, "DefaultDecoder")
        .def(py::init<int, int, bool>(), py::arg("hidden"), py::arg("output"), py::arg("is_continuous") = false);
    py::class_<G2048Decoder, std::shared_ptr<G2048Decoder>, Decoder>(m, "G2048Decoder")
        .def(py::init<int, int>());
    py::class_<NMMO3Encoder, std::shared_ptr<NMMO3Encoder>, Encoder>(m, "NMMO3Encoder")
        .def(py::init<int, int>());
    py::class_<NMMO3Decoder, std::shared_ptr<NMMO3Decoder>, Decoder>(m, "NMMO3Decoder")
        .def(py::init<int, int>());
    py::class_<DriveEncoder, std::shared_ptr<DriveEncoder>, Encoder>(m, "DriveEncoder")
        .def(py::init<int, int>());

    py::class_<RNN, std::shared_ptr<RNN>, torch::nn::Module>(m, "RNN");
    py::class_<MinGRU, std::shared_ptr<MinGRU>, RNN>(m, "MinGRU")
        .def(py::init<int, int, bool>(), py::arg("hidden"), py::arg("num_layers") = 1, py::arg("kernels") = true);

    py::class_<Policy, std::shared_ptr<Policy>, torch::nn::Module> cls2(m, "Policy");
    cls2.def(py::init<std::shared_ptr<Encoder>, std::shared_ptr<Decoder>, std::shared_ptr<RNN>, int, int, int>());
    cls2.def("forward", &Policy::forward);
    cls2.def("forward_train", &Policy::forward_train);
}
