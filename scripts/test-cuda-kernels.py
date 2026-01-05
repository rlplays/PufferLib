import torch
import pufferlib.native as nativelib



def test_sample_logits():
  B = 10
  A = 5
  logprobs = torch.zeros(B).cuda()
  actions = torch.zeros(B, A, dtype=torch.int64).cuda()
  nativelib.launch_sample_logits(torch.randn(B, A).cuda(),
                                torch.rand(B, A).cuda(),
                                logprobs,
                                actions
                                )
   
  print(f"Logprobs : {logprobs}")
  print(f"Actions : {actions}")


def test_linear_forward():
  R = 131
  C = 101
  CR = 31
  output = torch.zeros(R, C).cuda()
  nativelib.launch_linear_forward(torch.ones(R, CR).cuda(),
                                  torch.ones(C, CR).cuda(),
                                  torch.zeros(R, C).cuda(),
                                  output
                                  )
   
  print(f"Output : {output}")
