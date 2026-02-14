// advantage.cpp - VTrace/GAE/Puff advantage computation for pufferlib
// Separated from pufferlib.cpp for cleaner organization
// NOTE: This file is included directly into pufferlib.cpp inside namespace pufferlib

void puff_advantage_row(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}


void vtrace_check(Tensor values, Tensor rewards,
        Tensor dones, Tensor importance, Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}

// [num_steps, horizon]
void puff_advantage(float* values, float* rewards, float* dones, float* importance,
        float* advantages, float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, const int horizon){
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        puff_advantage_row(values + offset, rewards + offset,
            dones + offset, importance + offset, advantages + offset,
            gamma, lambda, rho_clip, c_clip, horizon
        );
    }
}

void compute_puff_advantage_cpu(Tensor values, Tensor rewards,
        Tensor dones, Tensor importance, Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check(values, rewards, dones, importance, advantages, num_steps, horizon);
    puff_advantage(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), importance.data_ptr<float>(), advantages.data_ptr<float>(),
        gamma, lambda, rho_clip, c_clip, num_steps, horizon
    );
}

// Forward declare CUDA implementation
void compute_puff_advantage_cuda(
    Tensor values,
    Tensor rewards,
    Tensor dones,
    Tensor importance,
    Tensor advantages,
    double gamma,
    double lambda,
    double rho_clip,
    double c_clip
);

