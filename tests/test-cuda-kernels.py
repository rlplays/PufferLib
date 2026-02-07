import torch
import pufferlib.native as nativelib



def test_sample_logits():
  T = 800
  A = 3
  N = 280
  logprobs = torch.zeros(T, 64).fill_(2.0).cuda().narrow(0, 3, N).select(1, 2)
  actions = torch.zeros(T, 64, A, dtype=torch.int64).fill_(1).cuda().narrow(0, 3, N).select(1, 2)
  logits = torch.randn(T, 64, 6).cuda().narrow(0, 3, N).select(1, 2)

  print(f"Before Actions : {actions.cpu()} {actions.stride()} {actions.shape}")
  print(f"Before Logprobs : {logprobs.cpu()} {logprobs.stride()} {logprobs.shape}")

  nativelib.launch_sample_logits_kernel(torch.randn(actions.shape).cuda(),
                                torch.tensor([2, 2, 2], dtype=torch.int64).cuda(),
                                torch.tensor([0, 2, 4], dtype=torch.int64).cuda(),
                                logits,
                                3,
                                actions,
                                logprobs)
  print(f"After Actions : {actions.cpu()}")
  print(f"After Logprobs : {logprobs.cpu()}")


test_sample_logits()



