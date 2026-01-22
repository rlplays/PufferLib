import pufferlib.ocean.breakout.binding as binding
import pufferlib.pytorch as puffypy
from pufferlib.pytorch import print_tensor, fill_sentinel, ensure_no_sentinel
import torch

def test_sample_logits():
  full_logits = torch.zeros(100, 64, 10).cuda()
  full_actions_out = torch.zeros(100, 64, dtype=torch.int32).cuda()
  full_logprobs_out = torch.zeros(100, 64).cuda()

  for j in range(10):
    val = j * 0.1 # Probability values [0, 1)
    full_logits.fill_(val)
    full_actions_out.fill_(2345) # Sentinel
    full_logprobs_out.fill_(-1.424242) # Sentinel
    torch.manual_seed(42)

    # Get C++ logprobs/actions
    logits = full_logits.select(1, 10)
    actions_out = full_actions_out.select(1, 10)
    logprobs_out = full_logprobs_out.select(1, 10)
    print(f'------------------------------- Iteration {j+1} -------------------------------')
    print_tensor(logits.cpu(), "Logits in", True)
    binding.sample_logits(logits, 1, [50], actions_out, logprobs_out)
    print_tensor(actions_out.cpu(), "C++ Actions Out", True)
    print_tensor(logprobs_out.cpu(), "C++ Logprobs Out", True)

    # CUDA version test in test_cuda_kernels.py
    # The test here only verifies that the raw (unused/existing) C++ version matches the PyTorch version.
    # The C++ version was converted to a CUDA kernel manually.

    # New PyTorch version test
    torch.manual_seed(42)
    actions1, logprobs1, _ = puffypy.sample_logits_v2(logits, 1, [50])
    print_tensor(actions1.cpu(), "Py Actions Out v2", True)
    print_tensor(logprobs1.cpu(), "Py Logprobs Out v2", True)
    verify_tensor = torch.eq(actions_out, actions1).all() and torch.allclose(logprobs_out, logprobs1)
    print(f"Verification : {verify_tensor}")
    assert(verify_tensor)

    # Old PyTorch version test
    torch.manual_seed(42)
    actions2, logprobs2, _ = puffypy.sample_logits(logits, 1, [50])
    print_tensor(actions2.cpu(), "Py Actions Out", True)
    print_tensor(logprobs2.cpu(), "Py Logprobs Out", True)
    verify_tensor = torch.eq(actions_out, actions2).all() and torch.allclose(logprobs_out, logprobs2)
    print(f"Verification : {verify_tensor}")
    assert(verify_tensor)


def test_sample_logits_entropy():
  full_logits = torch.zeros(100, 64, 10).cuda()
  full_actions_in = torch.zeros(100, 64, dtype=torch.int32).cuda()
  full_logprobs_out = torch.zeros(100, 64).cuda()
  full_entropy_out = torch.zeros(100, 64).cuda()

  for j in range(10):
    torch.manual_seed(42+j*23)
    if j == 0:
      full_actions_in.fill_(5) # Fixed action input for entropy calculation
      val = j * 0.1 # Probability values [0, 1)
      full_logits.fill_(val)
    else:
      full_actions_in.random_(0, 10) # Random actions
      full_logits.uniform_(0.0, 1.0) # Random logits
    full_logprobs_out.fill_(-1.424242) # Sentinel
    full_entropy_out.fill_(-1.424242) # Sentinel
    torch.manual_seed(42)

    # Get C++ logprobs/actions
    logits = full_logits.select(1, 10)
    actions_in = full_actions_in.select(1, 10)
    logprobs_out = full_logprobs_out.select(1, 10)
    entropy_out = full_entropy_out.select(1, 10)
    print(f'------------------------------- Iteration {j+1} -------------------------------')
    print_tensor(logits.cpu(), "Logits in", True)
    binding.sample_logits_with_entropy(logits, 1, [50], actions_in, logprobs_out, entropy_out)
    print_tensor(actions_in.cpu(), "C++ Actions Out", True)
    print_tensor(logprobs_out.cpu(), "C++ Logprobs Out", True)
    print_tensor(entropy_out.cpu(), "C++ Entropy Out", True)

    # New PyTorch version test
    torch.manual_seed(42)
    actions1, logprobs1, entropy1 = puffypy.sample_logits_v2(logits, 1, [50], actions_in)
    print_tensor(entropy1.cpu(), "Py Entropy Out v2", True)
    print_tensor(logprobs1.cpu(), "Py Logprobs Out v2", True)
    verify_tensor = torch.allclose(entropy1, entropy_out) and torch.allclose(logprobs_out, logprobs1)
    print(f"Verification : {verify_tensor}")
    assert(verify_tensor)

    # Old PyTorch version test
    torch.manual_seed(42)
    actions2, logprobs2, entropy2 = puffypy.sample_logits(logits, 1, [50], actions_in)
    print_tensor(entropy2.cpu(), "Py Entropy Out", True)
    print_tensor(logprobs2.cpu(), "Py Logprobs Out", True)
    verify_tensor = torch.allclose(entropy2, entropy_out) and torch.allclose(logprobs_out, logprobs2)
    print(f"Verification : {verify_tensor}")
    assert(verify_tensor)

test_sample_logits_entropy()
# test_sample_logits()
