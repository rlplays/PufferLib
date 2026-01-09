import pufferlib.ocean.go.binding as binding
import pufferlib.pytorch as puffypy
from pufferlib.pytorch import print_tensor, fill_sentinel, ensure_no_sentinel
import torch

def test_sample_logits():
  full_logits = torch.zeros(100, 64, 10).cuda()
  full_actions_out = torch.zeros(100, 64, dtype=torch.int32).cuda()
  full_logprobs_out = torch.zeros(100, 64).cuda()

  for j in range(10):
    val = j * 0.1
    full_logits.fill_(val)
    full_actions_out.fill_(2345)
    full_logprobs_out.fill_(-1.424242)
    torch.manual_seed(42)
    logits = full_logits.select(1, 10)
    actions_out = full_actions_out.select(1, 10)
    logprobs_out = full_logprobs_out.select(1, 10)
    print_tensor(logits.cpu(), "Logits in", True)
    binding.sample_logits(logits, 1, [50], actions_out, logprobs_out)
    print_tensor(actions_out.cpu(), "C++ Actions Out", True)
    print_tensor(logprobs_out.cpu(), "C++ Logprobs Out", True)

    torch.manual_seed(42)
    actions2, logprobs2, _ = puffypy.sample_logits(logits, 1, [50])
    print_tensor(actions2.cpu(), "Py Actions Out", True)
    print_tensor(logprobs2.cpu(), "Py Logprobs Out", True)
    verify_tensor = torch.eq(actions_out, actions2).all() and torch.allclose(logprobs_out, logprobs2)
    print(f"Verification : {verify_tensor}")
    assert(verify_tensor)

test_sample_logits()