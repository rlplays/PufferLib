import sys
from pdb import set_trace as T
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

import pufferlib
import pufferlib.models


numpy_to_torch_dtype_dict = {
    np.dtype("float64"): torch.float64,
    np.dtype("float32"): torch.float32,
    np.dtype("float16"): torch.float16,
    np.dtype("uint64"): torch.uint64,
    np.dtype("uint32"): torch.uint32,
    np.dtype("uint16"): torch.uint16,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int64"): torch.int64,
    np.dtype("int32"): torch.int32,
    np.dtype("int16"): torch.int16,
    np.dtype("int8"): torch.int8,
}


LITTLE_BYTE_ORDER = sys.byteorder == "little"

# USER NOTE: You should not get any errors in nativize.
# This is a complicated piece of code that attempts to convert
# flat bytes to structured tensors without breaking torch.compile.
# If you hit any errors, please post on discord.gg/puffer
# One exception: make sure you didn't change the dtype of your data
# ie by doing torch.Tensor(data) instead of torch.from_numpy(data)

# dtype of the tensor
# shape of the tensor
# starting element of the observation
# number of elements of the observation to take
# could be a namedtuple or dataclass
NativeDTypeValue = Tuple[torch.dtype, List[int], int, int]
NativeDType = Union[NativeDTypeValue, Dict[str, Union[NativeDTypeValue, "NativeDType"]]]

# TODO: handle discrete obs
# Spend some time trying to break this fn with differnt obs
def nativize_dtype(emulated) -> NativeDType:
    # sample dtype - the dtype of what we obtain from the environment (usually bytes)
    sample_dtype: np.dtype = emulated['observation_dtype']
    # structured dtype - the gym.Space converted numpy dtype

    # the observation represents (could be dict, tuple, box, etc.)
    structured_dtype: np.dtype = emulated['emulated_observation_dtype']
    subviews, dtype, shape, offset, delta = _nativize_dtype(sample_dtype, structured_dtype)
    if subviews is None:
        return (dtype, shape, offset, delta)
    else:
        return subviews

def round_to(x, base):
    return int(base * np.ceil(x/base))

def _nativize_dtype(sample_dtype: np.dtype,
        structured_dtype: np.dtype,
        offset: int = 0) -> NativeDType:
    if structured_dtype.fields is None:
        if structured_dtype.subdtype is not None:
            dtype, shape = structured_dtype.subdtype
        else:
            dtype = structured_dtype
            shape = (1,)

        delta = int(np.prod(shape))
        if sample_dtype.base.itemsize == 1:
            offset = round_to(offset, dtype.alignment)
            delta *= dtype.itemsize
        else:
            assert dtype.itemsize == sample_dtype.base.itemsize

        return None, numpy_to_torch_dtype_dict[dtype], shape, offset, delta
    else:
        subviews = {}
        start_offset = offset
        all_delta = 0
        for name, (dtype, _) in structured_dtype.fields.items():
            views, dtype, shape, offset, delta = _nativize_dtype(
                sample_dtype, dtype, offset)

            if views is not None:
                subviews[name] = views
            else:
                subviews[name] = (dtype, shape, offset, delta)

            offset += delta
            all_delta += delta

        return subviews, dtype, shape, start_offset, all_delta


def nativize_tensor(observation: torch.Tensor, native_dtype: NativeDType):
    return _nativize_tensor(observation, native_dtype)


# torch.view(dtype) does not compile
# This is a workaround hack
# @thatguy - can you figure out a more robust way to handle cast?
# I think it may screw up for non-uint data... so I put a hard .view
# fallback that breaks compile
def compilable_cast(u8, dtype):
    if dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        n = dtype.itemsize
        bytes = [u8[..., i::n].to(dtype) for i in range(n)]
        if not LITTLE_BYTE_ORDER:
            bytes = bytes[::-1]

        bytes = sum(bytes[i] << (i * 8) for i in range(n))
        return bytes.view(dtype)
    return u8.view(dtype)  # breaking cast


def _nativize_tensor(observation: torch.Tensor, native_dtype: NativeDType):
    if isinstance(native_dtype, tuple):
        dtype, shape, offset, delta = native_dtype
        torch._check_is_size(offset)
        torch._check_is_size(delta)
        # Important, we are assuming that obervations of shape
        # [N, D] where N is number of examples and D is number of
        # bytes per example is being passed in
        slice = observation.narrow(1, offset, delta)
        # slice = slice.contiguous()
        # slice = compilable_cast(slice, dtype)
        slice = slice.view(dtype)
        slice = slice.view(observation.shape[0], *shape)
        return slice
    else:
        subviews = {}
        for name, dtype in native_dtype.items():
            subviews[name] = _nativize_tensor(observation, dtype)
        return subviews


def nativize_observation(observation, emulated):
    # TODO: Any way to check that user has not accidentally cast data to float?
    # float is natively supported, but only if that is the actual correct type
    return nativize_tensor(
        observation,
        emulated['observation_dtype'],
        emulated['emulated_observation_dtype'],
    )

def flattened_tensor_size(native_dtype):
    return _flattened_tensor_size(native_dtype)

def _flattened_tensor_size(native_dtype):
    if isinstance(native_dtype, tuple):
        return np.prod(native_dtype[1])  # shape
    else:
        res = 0
        for _, dtype in native_dtype.items():
            res += _flattened_tensor_size(dtype)
        return res

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# taken from torch.distributions.Categorical
def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

# taken from torch.distributions.Categorical
def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    lp = logits_to_probs(logits)
    p_log_p = logits * lp
    return -p_log_p.sum(-1)

def entropy_probs(logits, probs):
    p_log_p = logits * probs
    return -p_log_p.sum(-1)

def sample_logits(logits, num_actions, action_nvec,action=None):
    is_discrete = num_actions == 1
    if isinstance(logits, torch.distributions.Normal):
        batch = logits.loc.shape[0]
        if action is None:
            mean = torch.nan_to_num(logits.loc, 0.0, 0.0, 0.0)
            std = torch.nan_to_num(logits.scale, 1.0, 1.0, 1.0)
            logits = torch.distributions.Normal(mean, std)
            action = logits.sample().view(batch, -1)

        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy
    elif is_discrete:
        logits = logits.unsqueeze(0)
    # TODO: Double check this
    else: #multi-discrete
        logits = logits.split(action_nvec, dim=1)       
        logits = torch.nn.utils.rnn.pad_sequence(
            [l.transpose(0,1) for l in logits], 
            batch_first=False, 
            padding_value=-torch.inf
        ).permute(1,2,0)

    # This can fail on nans etc.
    # Note: This is equivalent to torch.log_softmax. log(e^x_i / sum_j e^x_j) =  x_i - log(sum_j e^x_j) as x_i = log(e^x_i)
    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = logits_to_probs(logits)

    if action is None:
        probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
        action = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, replacement=True).int()
        action = action.reshape(probs.shape[:-1])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)
    logprob = log_prob(normalized_logits, action)
    logits_entropy = entropy(normalized_logits).sum(0)

    if is_discrete:
        return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)

    return action.T, logprob.sum(0), logits_entropy

def sample_logits_v2(logits, num_actions, action_nvec, action=None):
    if num_actions > 1: # multi-discrete, reshape from [N, sum(A_i * nvec_i)] to [N, num_actions, nvec_0]
        logits = logits.reshape(logits.shape[0], num_actions, action_nvec[0])
    logits = torch.nan_to_num(logits)
    logprobs = torch.log_softmax(logits, dim=-1)
    probs = logprobs.exp()
    if action is None:
        if num_actions > 1:
            probs = probs.reshape(-1, probs.shape[-1])
        action = torch.multinomial(probs, 1, replacement=True).int()
        logits_entropy = None
        if num_actions == 1:
            logprob = logprobs.gather(-1, action).squeeze(-1)
            action = action.squeeze(-1)
        else:
            action = action.squeeze().reshape(logits.shape[0], probs.shape[1])    
            logprob = logprobs.gather(-1, action.unsqueeze(-1)).squeeze(-1).sum(-1)
    else:
        batch = logits.shape[0]
        action = action.view(batch, -1)
        # Taken from torch.distributions.Categorical
        p_log_p = -(logprobs * probs).sum(-1)
        if num_actions > 1:
            p_log_p = p_log_p.sum(-1)
        else:
            p_log_p = p_log_p.squeeze(-1)
        logits_entropy = p_log_p

        if num_actions == 1:
            logprob = logprobs.gather(-1, action).squeeze(-1)
        else:
            logprob = logprobs.gather(-1, action.unsqueeze(-1)).squeeze(-1).sum(-1)


    return action, logprob, logits_entropy

def print_tensor(t, name, start = -50, N = None):
    flat = t.flatten().cpu()

    s_start = start
    s_stop = None
    if N is not None:
        s_stop = N if start is None else (start + N)
    to_print = flat[s_start:s_stop]
    extra = ""
    if t.dtype == torch.float32:
        extra = f" min={torch.min(t).item()} max={torch.max(t).item()} mean={torch.mean(t).item():.6f} std={torch.std(t).item():.6f}"
    print(
        f"{name}: shape={t.shape}, dtype={t.dtype}, stride={t.stride()} device={t.device} requires_grad={t.requires_grad} {extra}\n---------------------------------------------\n"
        + (f"S[{s_start}]" if start < 0 else "")
        + str(to_print.detach().numpy().tolist())
        + "\n---------------------------------------------\n"
    )

def print_gpu_mem(desc=""):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    print(f'GPU memory used {desc}: {used/1024/1024} MB / {total/1024/1024} MB')        

def fill_sentinel(t, sentinel=-1234):
    t.fill_(sentinel)

def ensure_no_sentinel(t, name, sentinel=-1234):
    t = t.cpu().detach().numpy()
    n_found = 0
    for v in t.flatten():
        if v == sentinel:
            n_found += 1
    if n_found > 0:
        print(f"Found {n_found} sentinel values in tensor {name}")

def compare_tensors(t1, t2, name):
    t1 = t1.flatten().float().cpu()
    t2 = t2.flatten().float().cpu()
    if t1.shape != t2.shape:
        print(f"{name}: Shapes differ: {t1.shape} vs {t2.shape}")
        return False
    error = 10e-3
    equal = torch.allclose(t1, t2, rtol=0.0, atol=error)
    if not equal:
        diff = (t1 - t2).abs()
        diffs = (diff > error).nonzero(as_tuple=False).flatten()
        print(f"{name}: Tensors differ at {len(diffs)} positions. First 10 diffs:")
        for i in range(min(10, len(diffs))):
            idx = diffs[i].item()
            print(f"---{name}: Index {idx}: t1={t1[idx]}, t2={t2[idx]}")
    return equal