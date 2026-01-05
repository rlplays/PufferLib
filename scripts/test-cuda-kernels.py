import torch
import pufferlib.native as nativelib



def test_sample_logits():
  N = 64
  A = 3
  logprobs = torch.zeros(N).fill_(42.0).cuda()
  actions = torch.zeros(N, A, dtype=torch.int64).fill_(123.0).cuda()
  logits = torch.randn(N*3).cuda()
  nativelib.launch_sample_logits(torch.randn(N, A).cuda(),
                                torch.from_numpy([2, 2, 2]).cuda(),
                                torch.from_numpy([0, 2, 4]).cuda(),
                                logits,
                                
                                logprobs,
                                actions
                                )
   
  print(f"Logprobs : {logprobs}")
  print(f"Actions : {actions}")


test_sample_logits()
