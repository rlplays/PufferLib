#ifndef PUFFERLIB_MODULES_H
#define PUFFERLIB_MODULES_H

#include <torch/extension.h>
#include <torch/torch.h>

// Loss component indices for the shared accumulator tensor
enum LossIdx {
    LOSS_PG = 0,
    LOSS_VF = 1,
    LOSS_ENT = 2,
    LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4,
    LOSS_APPROX_KL = 5,
    LOSS_CLIPFRAC = 6,
    LOSS_N = 7,           // number of accumulations (also == number of loss components)
    NUM_LOSSES = 8,
};

// CUDA kernel wrappers (implemented in modules.cu)

// Fused mingru gate inference
std::vector<torch::Tensor> mingru_gate(torch::Tensor state, torch::Tensor combined);

// Fused scan with checkpointing (training path)
torch::autograd::tensor_list fused_scan_checkpointed(torch::Tensor combined, torch::Tensor state);

// LogCumSumExp
torch::Tensor logcumsumexp_cuda(torch::Tensor x);

// Fused PPO loss (optimized kernel version)
torch::autograd::tensor_list fused_ppo_loss_optimized(
    torch::Tensor logits,
    torch::Tensor logstd,
    torch::Tensor values_pred,
    torch::Tensor actions,
    torch::Tensor old_logprobs,
    torch::Tensor advantages,
    torch::Tensor prio,
    torch::Tensor values,
    torch::Tensor returns,
    torch::Tensor adv_mean,
    torch::Tensor adv_var,
    torch::Tensor ratio_out,
    torch::Tensor newvalue_out,
    torch::Tensor act_sizes,
    torch::Tensor losses_acc,
    float clip_coef,
    float vf_clip_coef,
    float vf_coef,
    float ent_coef
);

// Fused sample_logits: handles both discrete and continuous action sampling
void sample_logits(
    torch::Tensor logits,
    torch::Tensor logstd,
    torch::Tensor value,
    torch::Tensor actions_out,
    torch::Tensor logprobs_out,
    torch::Tensor value_out,
    torch::Tensor act_sizes,
    uint64_t seed,
    torch::Tensor offset
);

// FCMax: fused FC -> Max
torch::Tensor fc_max(torch::Tensor x, torch::Tensor W, torch::Tensor b);

// Compute prio: fused priority sampling for minibatch selection
std::tuple<torch::Tensor, torch::Tensor> compute_prio_cuda(
    torch::Tensor advantages, float prio_alpha,
    int minibatch_segments, int total_agents, float anneal_beta);

// Select + Copy: fused index_select and copy for minibatch preparation
void train_select_and_copy_cuda(
    torch::Tensor observations, torch::Tensor actions,
    torch::Tensor logprobs, torch::Tensor values, torch::Tensor advantages,
    torch::Tensor idx, torch::Tensor mb_prio,
    torch::Tensor dst_obs, torch::Tensor dst_state,
    torch::Tensor dst_actions, torch::Tensor dst_logprobs,
    torch::Tensor dst_advantages, torch::Tensor dst_prio,
    torch::Tensor dst_values, torch::Tensor dst_returns);

// Puff Advantage CUDA dispatch (in namespace pufferlib)
namespace pufferlib {
void compute_puff_advantage_cuda(
    torch::Tensor values, torch::Tensor rewards,
    torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
    double gamma, double lambda, double rho_clip, double c_clip);
}

#endif // PUFFERLIB_MODULES_H
