#pragma once

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>

static constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

inline torch::Tensor zeropower_via_newtonschulz(torch::Tensor G) {
    auto x = G.to(torch::kBFloat16);
    if (G.size(-2) > G.size(-1)) {
        x = x.mT();
    }

    x.div_(x.norm().clamp(1e-7));

    for (int i = 0; i < 5; ++i) {
        auto a = ns_coeffs[i][0];
        auto b = ns_coeffs[i][1];
        auto c = ns_coeffs[i][2];
        auto A = x.mm(x.mT());
        auto gram_update = at::addmm(A, A, A, b, c);
        x = at::addmm(x, gram_update, x, a, 1.0);
    }

    if (G.size(-2) > G.size(-1)) {
        x = x.mT();
    }

    return x.to(G.dtype());
}

struct Muon {
    // Hyperparameters
    double momentum;
    double weight_decay;
    double eps;

    // State
    torch::Tensor lr;              // scalar CUDA tensor
    torch::Tensor weight_buffer;   // contiguous fp32 param buffer
    torch::Tensor momentum_buffer; // contiguous momentum buffer
    std::vector<torch::Tensor> params;

    // Multi-GPU
    ncclComm_t nccl_comm = nullptr;
    int world_size = 1;

    Muon(std::vector<torch::Tensor> params, double lr_val, double momentum,
         double eps, double weight_decay)
        : momentum(momentum), weight_decay(weight_decay), eps(eps),
          params(std::move(params))
    {
        TORCH_CHECK(lr_val >= 0, "Invalid learning rate: ", lr_val);
        TORCH_CHECK(eps >= 0, "Invalid epsilon value: ", eps);
        TORCH_CHECK(weight_decay >= 0, "Invalid weight_decay value: ", weight_decay);
        lr = torch::tensor(lr_val, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    }

    void init_contiguous_weights() {
        torch::NoGradGuard no_grad;

        int64_t total_size = 0;
        for (auto& p : params) {
            total_size += p.numel();
        }

        auto device = params[0].device();
        weight_buffer = torch::zeros({total_size},
            torch::dtype(torch::kFloat32).device(device));
        weight_buffer.set_requires_grad(true);

        int64_t offset = 0;
        for (auto& p : params) {
            int64_t size = p.numel();
            auto shape = p.sizes().vec();
            weight_buffer.narrow(0, offset, size).copy_(p.flatten());
            torch::Tensor view = weight_buffer.narrow(0, offset, size).view(shape);
            p.set_data(view);
            offset += size;
        }
    }

    void step() {
        torch::NoGradGuard no_grad;

        if (!weight_buffer.defined()) return;

        // Initialize momentum buffer lazily
        if (!momentum_buffer.defined()) {
            momentum_buffer = torch::zeros_like(weight_buffer);
        }

        // Gather grads into contiguous buffer
        torch::Tensor all_grads = torch::zeros_like(weight_buffer);
        int64_t offset = 0;
        for (auto& p : params) {
            int64_t size = p.numel();
            if (p.grad().defined()) {
                all_grads.narrow(0, offset, size).copy_(p.grad().flatten());
            }
            offset += size;
        }

        // Multi-GPU gradient sync
        if (nccl_comm != nullptr && world_size > 1) {
            ncclAllReduce(all_grads.data_ptr(), all_grads.data_ptr(),
                          all_grads.numel(), ncclFloat, ncclAvg,
                          nccl_comm, at::cuda::getCurrentCUDAStream());
        }

        // Nesterov momentum
        momentum_buffer.mul_(momentum);
        momentum_buffer.add_(all_grads);
        all_grads.add_(momentum_buffer, momentum);

        // Newton-Schulz per param
        torch::Tensor all_updates = torch::zeros_like(weight_buffer);
        offset = 0;
        for (auto& p : params) {
            int64_t size = p.numel();
            if (p.grad().defined()) {
                torch::Tensor update = all_grads.narrow(0, offset, size).view(p.sizes());
                if (p.dim() >= 2) {
                    auto G = update.view({update.size(0), -1});
                    update = zeropower_via_newtonschulz(G);
                    double ratio = (double)update.size(-2) / (double)update.size(-1);
                    double scale = std::sqrt(std::max(1.0, ratio));
                    update.mul_(scale);
                }
                all_updates.narrow(0, offset, size).copy_(update.flatten());
            }
            offset += size;
        }

        // Apply update
        if (weight_decay != 0) {
            weight_buffer.mul_(1 - lr * weight_decay);
        }
        weight_buffer.sub_(all_updates * lr);
    }

    void zero_grad() {
        for (auto& p : params) {
            if (p.grad().defined()) {
                p.grad().zero_();
            }
        }
    }

    std::unordered_map<std::string, torch::Tensor> state_dict() const {
        std::unordered_map<std::string, torch::Tensor> state;
        state["lr"] = lr;
        if (weight_buffer.defined()) state["weight_buffer"] = weight_buffer;
        if (momentum_buffer.defined()) state["momentum_buffer"] = momentum_buffer;
        return state;
    }

    void load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state) {
        auto it = state.find("lr");
        if (it != state.end()) lr.copy_(it->second);
        it = state.find("weight_buffer");
        if (it != state.end()) weight_buffer.copy_(it->second);
        it = state.find("momentum_buffer");
        if (it != state.end()) momentum_buffer.copy_(it->second);
    }
};
