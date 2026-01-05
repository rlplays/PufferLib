import torch
import pufferlib.native as nativelib



def test_sample_logits():
  N = 42
  A = 3
  logprobs = torch.zeros(N).fill_(2.0).cuda()
  actions = torch.zeros(N, A, dtype=torch.int64).fill_(1).cuda()
  logits = torch.randn(N, 6).cuda()
  nativelib.launch_sample_logits_kernel(torch.randn(N, A).cuda(),
                                torch.tensor([2, 2, 2], dtype=torch.int64).cuda(),
                                torch.tensor([0, 2, 4], dtype=torch.int64).cuda(),
                                logits,
                                3,
                                actions,
                                logprobs)
   
  print(f"Actions : {actions.cpu()}")
  print(f"Logprobs : {logprobs.cpu()}")


test_sample_logits()



