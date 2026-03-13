## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
# TODO Evaluate this # We need this option as we allocate large chunks of memory for multiple envs across many threads.
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
# os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

import sys
import glob
import ast
import time
import random
import shutil
import subprocess
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension
import torch.profiler
import torch.cuda._memory_viz

import pickle

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
from pufferlib.pytorch import print_tensor, print_gpu_mem, compare_tensors
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
import rich.pretty
import pprint
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME
)
# Assume advantage kernel has been built if torch has been compiled with CUDA or HIP support
# and can find CUDA or HIP in the system
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)

class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        # Backend perf optimization
        # torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.fp32_precision = "tf32"

        torch.backends.cudnn.deterministic = config['torch_deterministic']
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.allow_tf32 = True
        # torch.cuda.set_sync_debug_mode(0)

        # Reproducibility
        seed = config['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        # Experience
        if config['batch_size'] == 'auto' and config['bptt_horizon'] == 'auto':
            raise pufferlib.APIUsageError('Must specify batch_size or bptt_horizon')
        elif config['batch_size'] == 'auto':
            config['batch_size'] = total_agents * config['bptt_horizon']
        elif config['bptt_horizon'] == 'auto':
            config['bptt_horizon'] = config['batch_size'] // total_agents

        batch_size = config['batch_size']
        horizon = config['bptt_horizon']
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f'Total agents {total_agents} <= segments {segments}'
            )

        device = config['device']

        # Native libtorch + multithreading
        self.use_native_libtorch = \
          hasattr(vecenv, 'enable_native_libtorch') and vecenv.enable_native_libtorch and \
          policy.support_native_libtorch()
        self.use_native_libtorch_train = \
          self.use_native_libtorch and \
          hasattr(vecenv, 'enable_native_libtorch_train') and vecenv.enable_native_libtorch_train

        if self.use_native_libtorch:
            # Native libtorh requires float32 observations and int64 actions.
            self.observations = torch.zeros(segments, horizon, *obs_space.shape,
              dtype=torch.float32,
              pin_memory=device == 'cuda' and config['cpu_offload'],
              device=device)
            # Native libtorch converts the actions to the corresponding internal type manually.
            self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
              dtype=torch.int32)
        else:          
            self.observations = torch.zeros(segments, horizon, *obs_space.shape,
              dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
              pin_memory=device == 'cuda' and config['cpu_offload'],
              device='cpu' if config['cpu_offload'] else device)
            self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
              dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])

        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents

        # Native libtorch training setup
        if self.use_native_libtorch_train:
            vecenvs = vecenv.get_vecenvs()
            binding = vecenv.get_binding()
            train_opts = binding.PufferTrainOpts()
            cfg = dict(train_opts.config)
            for k, v in config.items():
                cfg[k] = str(v)
            train_opts.config = cfg            
            binding.torch_init_train_lstm(vecenvs, train_opts)

        # LSTM
        if config['use_rnn']:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
            self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}

        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']
        max_minibatch_size = config['max_minibatch_size']
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f'minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly')

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(
                f'batch_size {batch_size} must be >= minibatch_size {minibatch_size}'
            )

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config['update_epochs'] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon 
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f'minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}'
            )

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        policy.policy.use_native_libtorch = self.use_native_libtorch

        if config['compile'] and self.use_native_libtorch and not self.use_native_libtorch_train:
            self.policy = torch.compile(policy, mode=config['compile_mode'])
            self.policy.forward_eval = torch.compile(policy.forward_eval, mode=config['compile_mode'])
            pufferlib.pytorch.sample_logits = torch.compile(pufferlib.pytorch.sample_logits, mode=config['compile_mode'])

        # Optimizer
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
            )
        elif config['optimizer'] == 'muon':
            import heavyball
            from heavyball import ForeachMuon
            warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
            heavyball.utils.compile_mode = config.get('compile_mode', 'reduce-overhead')

            # # optionally a little bit better/faster alternative to newtonschulz iteration
            # import heavyball.utils
            # heavyball.utils.zeroth_power_mode = 'thinky_polar_express'

            # heavyball_momentum=True introduced in heavyball 2.1.1
            # recovers heavyball-1.7.2 behaviour - previously swept hyperparameters work well
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
                heavyball_momentum=True,
            )
        else:
            raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

        self.optimizer = optimizer

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Profile (perf metrics)
        self.profile_info = None

        # Learning rate scheduler
        epochs = config['total_timesteps'] // config['batch_size']
        eta_min = config['learning_rate'] * config['min_lr_ratio']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config['precision']
        self.amp_context = contextlib.nullcontext()
        if config.get('amp', True) and config['device'] == 'cuda':
            self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
        if precision not in ('float32', 'bfloat16'):
            raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    @torch.no_grad()
    def evaluate(self):
        if self.use_native_libtorch:
            stats = self.evaluate_native()
        else:
            stats = self.evaluate_python()
        return stats

    def evaluate_native(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        config = self.config
        device = config['device']

        # self.print_gpu_mem("Before setup")
        self.policy.setup_native_libtorch_eval(self.vecenv, self.observations, self.actions, 
                                               self.logprobs, self.rewards, self.terminals, self.values)
        # Runs the entire horizon and obtains the results provided during setup above.
        # self.print_gpu_mem("After setup")
        self.policy.run_native_libtorch_eval(self.vecenv)
        # for segment in range(0, 64):
        #   if hasattr(self.vecenv, 'get_binding'):
        #     o, r, d, t, info, env_id, mask = self.vecenv.recv()
        #     batch = 0
        #     self.vecenv.get_binding().torch_run_single_eval(self.vecenv.get_vecenvs(), batch)
        #     a_copy = self.actions.select(1, segment)
        #     self.vecenv.send(a_copy.cpu().numpy())

        # self.print_gpu_mem("After run")

        # Returns the stats collected during evaluation.
        (info, eval_result) = self.policy.finish_native_libtorch_eval(self.vecenv)

        # self.print_gpu_mem("After finish")
        # rich.pretty.pprint(dict(eval_result.stats_millis))
        s = {stat.name: stat for stat in eval_result.perf_stats}
        # for k in s:
        #     print(f"{k}: total_duration_ms={s[k].total_duration_ms} num_batches={s[k].num_batches}")
        # eval_copy/eval_forward are averaged from across different threads/batches in C++ to
        # present a fake wall-clock time so that Train vs Eval can be compared.
        # The stats do have a _sum version which is the total (overlapping) time spent across threads/batches.
        profile.add('eval_copy', epoch, (s['to_device_copy'].total_duration_ms) / (1000.0 * s['to_device_copy'].num_batches))
        profile.add('eval_forward', epoch, s['lstm_forward'].total_duration_ms / (1000.0 * s['lstm_forward'].num_batches))
        profile.add('env', epoch, s['env_cpu'].total_duration_ms / (1000.0 * s['env_cpu'].num_batches))
        self.global_step += eval_result.step_count

        self.profile_info = s
        self.profile_info['eval_steps'] = eval_result.step_count
        for k, v in pufferlib.unroll_nested_dict(info):
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, (list, tuple)):
                self.stats[k].extend(v)
            else:
                self.stats[k].append(v)        
        # for i in info:
        #     for k, v in pufferlib.unroll_nested_dict(i):
        #         if isinstance(v, np.ndarray):
        #             v = v.tolist()
        #         elif isinstance(v, (list, tuple)):
        #             self.stats[k].extend(v)
        #         else:
        #             self.stats[k].append(v)
        return self.stats

    def evaluate_python(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']

        if config['use_rnn']:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        self.full_rows = 0

        while self.full_rows < self.segments:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t # TODO: Handle truncations separately
            self.global_step += int(mask.sum())

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)#, non_blocking=True)
            r = torch.as_tensor(r).to(device)#, non_blocking=True)
            d = torch.as_tensor(d).to(device)#, non_blocking=True)

            profile('eval_forward', epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config['use_rnn']:
                    state['lstm_h'] = self.lstm_h[env_id.start]
                    state['lstm_c'] = self.lstm_c[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = self.policy.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

            profile('eval_copy', epoch)
            with torch.no_grad():
                if config['use_rnn']:
                    self.lstm_h[env_id.start] = state['lstm_h']
                    self.lstm_c[env_id.start] = state['lstm_c']

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1+self.ep_indices[env_id.stop - 1].item())

                if config['cpu_offload']:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l+1 >= config['bptt_horizon']:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config['device']).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile('eval_misc', epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile('env', epoch)
            self.vecenv.send(action)

        profile('eval_misc', epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    @record
    def train(self):
        if self.use_native_libtorch_train:
            logs = self.train_native()
        else:
            logs = self.train_python()
        return logs

    def print_weights(self, weights, name = ""):
        print_tensor(weights.encoder_w, f"encoder_w {name}", 0, 10)
        print_tensor(weights.encoder_b, f"encoder_b {name}", 0, 10)
        print_tensor(weights.decoder_w, f"decoder_w {name}", 0, 10)
        print_tensor(weights.decoder_b, f"decoder_b {name}", 0, 10)
        print_tensor(weights.value_w, f"value_w {name}", 0, 10)
        print_tensor(weights.value_b, f"value_b {name}", 0, 10)
        print_tensor(weights.lstm_weight_ih, f"lstm_weight_ih {name}", 0, 10)
        print_tensor(weights.lstm_weight_hh, f"lstm_weight_hh {name}", 0, 10)
        print_tensor(weights.lstm_bias_ih, f"lstm_bias_ih {name}", 0, 10)
        print_tensor(weights.lstm_bias_hh, f"lstm_bias_hh {name}", 0, 10)


    def train_native(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        config = self.config

        vecenvs = self.vecenv.get_vecenvs()
        binding = self.vecenv.get_binding()

        should_train = True # To quickly test eval vs eval+train
        if should_train:
          result = binding.torch_train_lstm(
              vecenvs, int(self.epoch), int(self.total_epochs), int(self.segments), int(self.total_minibatches), int(self.minibatch_segments), int(self.accumulate_minibatches),
              self.observations, self.actions, self.logprobs, self.rewards, self.terminals, self.values)
          losses = {result.name: result.value_dbl for result in result.train_stats}
        else:
          losses = {}
        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.finalize_weights()
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs        


    @record
    def train_python(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        profile('train_misc', epoch, nest=True)
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1
        for mb in range(self.total_minibatches):
            profile('train_misc', epoch)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(self.values, self.rewards,
                self.terminals, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])
            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta

            profile('train_copy', epoch)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]
            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )
            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = self.policy.sample_logits(logits, action=mb_actions)

            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            # NOTE: Commenting this out since adv is replaced below
            # adv = advantages[idx]
            # adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
            #     ratio, adv, config['gamma'], config['gae_lambda'],
            #     config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Weight advantages by priority and normalize
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
            self.amp_context.__enter__() # TODO: AMP needs some debugging

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()

            # Logging
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.item() / self.total_minibatches
            losses['value_loss'] += v_loss.item() / self.total_minibatches
            losses['entropy'] += entropy_loss.item() / self.total_minibatches
            losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
            losses['approx_kl'] += approx_kl.item() / self.total_minibatches
            losses['clipfrac'] += clipfrac.item() / self.total_minibatches
            losses['importance'] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile('learn', epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs

    def finalize_weights(self):
        if self.use_native_libtorch and self.use_native_libtorch_train:
          # Transfer weights from native libtorch training to PyTorch model for serialization.
          vecenvs = self.vecenv.get_vecenvs()
          binding = self.vecenv.get_binding()
          weights = binding.torch_train_get_weights(vecenvs)
          self.policy.policy.encoder[0].weight.data = weights.encoder_w
          self.policy.policy.encoder[0].bias.data = weights.encoder_b
          self.policy.policy.decoder.weight.data = weights.decoder_w
          self.policy.policy.decoder.bias.data = weights.decoder_b
          self.policy.policy.value.weight.data = weights.value_w
          self.policy.policy.value.bias.data = weights.value_b
          self.policy.lstm.weight_ih_l0.data = weights.lstm_weight_ih
          self.policy.lstm.weight_hh_l0.data = weights.lstm_weight_hh
          self.policy.lstm.bias_ih_l0.data = weights.lstm_bias_ih
          self.policy.lstm.bias_hh_l0.data = weights.lstm_bias_hh
          print("...Finalized weights from native libtorch training to PyTorch model.")

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config['device']
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            'SPS': dist_sum(self.sps, device),
            'agent_steps': agent_steps,
            'uptime': time.time() - self.start_time,
            'epoch': int(dist_sum(self.epoch, device)),
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v['elapsed'] for k, v in self.profile},
            #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                self.logger.log(logs, agent_steps)
                return logs
            else:
                return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[dim default]', b1='[bright_cyan]', b2='[default]'):
        # return None
        config = self.config
        sps = dist_sum(self.sps, config['device'])
        agent_steps = dist_sum(self.global_step, config['device'])
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True,
            show_header=False, border_style='bright_cyan')
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f'{b1}PufferLib {b2}3.0_Peru_RLPlays {idx[0]*" "}:blowfish: ',
            f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
            f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
            f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
            f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
        )
        idx[0] = (idx[0] - 1) % 10

        s = Table(box=None, expand=True)
        remaining = f'{b2}A hair past a freckle{c2}'
        if sps != 0:
            remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{b2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{b2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{b2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{b2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{b2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{b2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{b2}Remaining', remaining)

        delta = profile.eval['buffer'] + profile.train['buffer']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        suffix = ''
        if self.use_native_libtorch:
            suffix = '(approx/MT)'
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf(f'  Forward {suffix}', b2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf(f'  Env {suffix}', b2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf(f'  Copy {suffix}', b2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf(f'  Misc {suffix}', b2, delta, profile.eval_misc, b2, c2))
        suffix = ''
        if self.use_native_libtorch_train:
            suffix = '(approx/MT)'
        p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf(f'  Forward {suffix}', b2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf(f'  Learn {suffix}', b2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf(f'  Copy {suffix}', b2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf(f'  Misc {suffix}', b2, delta, profile.train_misc, b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version.'''

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages

def abbreviate(num, b2, c2):
    if num < 1e3:
        return f'{b2}{num}{c2}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}K'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}M'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}B'
    else:
        return f'{b2}{num/1e12:.2f}{c2}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()

class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]['start'] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile['start']
        profile['delta'] += delta
        # Multiply delta by freq to account for skipped epochs
        profile['elapsed'] += delta * self.frequency

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def add(self, name, epoch, elapsed):
        if (epoch + 1) % self.frequency != 0:
            return
        profile = self.profiles[name]
        profile['delta'] += elapsed
        profile['elapsed'] += elapsed * self.frequency

    def clear(self):
        for prof in self.profiles.values():
            if prof['delta'] > 0:
                prof['buffer'] = prof['delta']
                prof['delta'] = 0

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                   time.sleep(self.delay)
                   continue

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100*(total-free)/total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]

class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100*time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path, early_stop):
        pass

class NeptuneLogger:
    def __init__(self, args, load_id=None, mode='async'):
        import neptune as nept
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune['model'].track_files(model_path)

    def close(self, model_path, early_stop):
        self.neptune['early_stop'] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination='artifacts')
        return f'artifacts/{self.run_id}.pt'

class WandbLogger:
    def __init__(self, args, load_id=None, resume='allow'):
        import wandb
        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args['wandb_project'],
            group=args['wandb_group'],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags = [args['tag']] if args['tag'] is not None else [],
            settings=wandb.Settings(console="off"),  # stop sending dashboard to wandb
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path, early_stop):
        self.wandb.run.summary['early_stop'] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'
def train(env_name, args=None, vecenv=None, policy=None, logger=None, early_stop_fn=None):
    # If args is not provided, load config from config/default.ini and override with provided config/<env_name>.ini
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print("World size", world_size)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if 'LOCAL_RANK' in os.environ:
        args['train']['device'] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend='nccl', world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, 'lstm'):
            #model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)

    train_config = { **args['train'], 'env': env_name }
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    # Sweep needs data for early stopped runs, so send data when steps > 100M
    logging_threshold = min(0.20*train_config['total_timesteps'], 100_000_000)
    all_logs = []

    while pufferl.global_step < train_config['total_timesteps']:
        pufferl.evaluate()
        logs = pufferl.train()

        if logs is not None:
            should_stop_early = False
            if early_stop_fn is not None:
                should_stop_early = early_stop_fn(logs)
                # This is hacky, but need to see if threshold looks reasonable
                if 'early_stop_threshold' in logs:
                    pufferl.logger.log({'environment/early_stop_threshold': logs['early_stop_threshold']}, logs['agent_steps'])

            if pufferl.global_step > logging_threshold:
                all_logs.append(logs)

            if should_stop_early:
                model_path = pufferl.close()
                pufferl.logger.close(model_path, early_stop=True)
                return all_logs

    print("Final eval")
    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    for i in range(128):  # Run eval for at least 32, but put a hard stop at 128.
        stats = pufferl.evaluate()
        if i >= 32 and stats:
            break

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    pufferl.finalize_weights()
    print(f"Starting model save:")
    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    print(f"...Model saved to {model_path}")
    return all_logs

def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args['vec']['backend']
    if backend != 'PufferEnv':
        backend = 'Serial'

    args['vec'] = dict(backend=backend, num_envs=1)
    args['vec']['enable_native_libtorch'] = 0
    args['vec']['enable_native_libtorch_train'] = 0
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args['train']['device']

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    while True:
        render = driver.render()
        if len(frames) < args['save_frames']:
            frames.append(render)

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1/args['fps'])
        elif driver.render_mode == 'rgb_array':
            pass
            #import cv2
            #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #cv2.imshow('frame', render)
            #cv2.waitKey(1)
            #time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = policy.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if len(frames) > 0 and len(frames) == args['save_frames']:
            import imageio
            imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
            print(f'Saved {len(frames)} frames to {args["gif_path"]}')

def stop_if_loss_nan(logs):
    return any("losses/" in k and np.isnan(v) for k, v in logs.items())

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args['wandb'] and not args['neptune']:
        raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')
    args['no_model_upload'] = True  # Uploading trained model during sweep crashed wandb

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'
    running_target_buffer = deque(maxlen=30)

    def stop_if_perf_below(logs):
        if stop_if_loss_nan(logs):
            logs['is_loss_nan'] = True
            return True

        if method != 'Protein':
            return False

        if ('uptime' in logs and target_key in logs):
            metric_val, cost = logs[target_key], logs['uptime']
            running_target_buffer.append(metric_val)
            target_running_mean = np.mean(running_target_buffer)
            
            # If metric distribution is percentile, threshold is also logit transformed
            threshold = sweep.get_early_stop_threshold(cost)
            logs['early_stop_threshold'] = max(threshold, -5)  # clipping for visualization

            if sweep.should_stop(max(target_running_mean, metric_val), cost):
                logs['is_loss_nan'] = False
                return True
        return False

    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # In the first run, skip sweep and use the train args specified in the config
        if i > 0:
            sweep.suggest(args)


        all_logs = train(env_name, args=args, early_stop_fn=stop_if_perf_below)
        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

        is_final_loss_nan = all_logs[-1].get('is_loss_nan', False)
        if is_final_loss_nan:
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def profile(args_in=None, env_name=None, vecenv_in=None, policy_in=None):
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    profile_txt = f'----Start profiling results {env_name} {ts}----\n\n'

    args = args_in or load_config(env_name)
    cuda_trace_enabled = args['profile']['trace']
    do_eval = args['profile']['eval'] != 0
    do_train = args['profile']['train'] != 0
    profile_type = f'{do_eval*"eval_"}{do_train*"train_"}'
    profile_name = f'_{profile_type}_{args["profile"]["name"]}' if args["profile"]["name"] else ''
    args['env_name'] = env_name
    vecenv = vecenv_in or load_env(env_name, args)
    policy = policy_in or load_policy(args, vecenv)
    logger = None
    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)

    train_config = { **args['train'], 'env': env_name }
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    # Warmup
    for _ in range(5):
        if do_eval:
            pufferl.evaluate()
        if do_train:
            pufferl.train()

    # Conditionally enable memory recording
    enable_memory_profile = (args["profile"]["memory"] != 0)
    N = 10

    if enable_memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=100000, context='all')
        N = 1  # Memory profiling is slow, do only one run
    # Raw timing
    s0 = pufferl.global_step
    t0 = time.perf_counter()        
    memory_context = torch.profiler.record_function("evaluate") if not enable_memory_profile else contextlib.nullcontext()
    stats = None
    with memory_context:
        for _ in range(N):
          if do_eval:
              stats = pufferl.evaluate()
          if do_train:
              pufferl.train()
    t1 = time.perf_counter()
    s1 = pufferl.global_step
    diff_steps = s1 - s0
    diff = t1 - t0

    # Only capture snapshot if memory profiling was enabled
    if enable_memory_profile:
        snapshot = torch.cuda.memory._snapshot()
        mem_snapshot_name = f"experiments/memsnapshot{profile_name}{ts}.pickle"
        with open(mem_snapshot_name, 'wb') as f:
            pickle.dump(snapshot, f)
        
        torch.cuda.memory._record_memory_history(enabled=None)
        html_filename = f"experiments/memtrace{profile_name}{ts}.html"
        subprocess.run([
            sys.executable, '-m', 'torch.cuda._memory_viz', 
            'trace_plot', mem_snapshot_name, '-o', html_filename
        ])        
        print(f"Memory snapshot HTML saved to {html_filename}")
        os._exit(0)
        
    txt = ""
    if stats is not None:
        profile_txt += pprint.pformat(stats) + "\n\n"
    if pufferl.profile_info is not None:
      for k, v in pufferl.profile_info.items():
        if k not in ['eval_steps', 'total_forward_eval']:
          txt += f'--- {k} ---\n'
          for attr in dir(v):
              if not attr.startswith('_'):
                  try:
                      value = getattr(v, attr)
                      txt += f"------  {attr}: {value}\n"
                  except Exception as e:
                      print(f"{attr}: <error: {e}>")        

    txt += f"{env_name}:{profile_name} took {diff:.3f} seconds / {N} runs = {diff/N:.3f} seconds per run\n"
    txt += f"   - {env_name}{profile_name} {diff_steps} steps evaluated. SPS: {diff_steps/diff:.3f}\n"
    profile_txt += f'----------- Profile for {env_name}{profile_name} -----------\n'
    profile_txt += txt + '\n'

    # Capture CUDA trace that you can view with ui.perfetto.dev.
    if cuda_trace_enabled == 1:
        print("Now capturing CUDA trace. This may take a while...")
        trace_file = f'experiments/torchtrace_{ts}_{args['env_name']}_{profile_name}.json'
        import torchvision.models as models
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                     record_shapes=True, profile_memory = True, with_stack=True) as prof:
            with record_function("model_inference"):
                for i in range(5):
                    print("Profiling iteration", i+1)
                    if do_eval:
                        pufferl.evaluate()
                    if do_train:
                        pufferl.train()
        print(f"Profiling completed. Exporting to trace file {trace_file}...")
        perf_results = prof.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total', row_limit=50)
        print(perf_results)
        profile_txt += perf_results + '\n'
        prof.export_chrome_trace(trace_file)
        print(f'Exported trace to {trace_file}')
        profile_txt += f'Profile for {env_name} {profile_name} (full trace in {trace_file}):\n{perf_results}\n\n'
        print(profile_txt)

        
    profile_txt += f'----------- Completed profile for {env_name}{profile_name} -----------\n'
    vecenv.close()
    vecenv = None

    text_file = f'experiments/torchtrace_{ts}_{args['env_name']}_{profile_name}.txt'
    with open(text_file, 'w') as f:
        f.write(profile_txt)      

    print(txt)
    print(f'Exported perf data to {text_file}')
    os._exit(0)


def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    # Update vec config instead of replacing it
    args['vec']['backend'] = 'Serial'
    args['vec']['num_envs'] = 1
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)

    def output_config_():
        config_str = f"weights={path}\n"
        config_str += f"num_weights={len(weights)}\n"
        sections = ['env', 'vec', 'rnn', 'policy', 'base']
        for section in sections:
            env_args = args[section]
            if env_args is not None:
                for k, v in env_args.items():
                    print(f"Adding config {section}.{k}={v}")
                    config_str += f"{section}.{k}={v}\n"
        return config_str
        
    
    target_name = env_name.replace('puffer_', '')
    if (target_name != env_name):
        path = f'resources/{target_name}/{target_name}_weights.bin'
        weights.tofile(path)
        config_str = output_config_()

        # Write config to resources/<env_name>/<env_name>_config.ini
        # Contains the weights count+path and env args

        with open(f'resources/{target_name}/{target_name}_config.ini', 'w') as f:
            f.write(config_str)
            
        print(f'Config written to resources/{target_name}/{target_name}_config.ini')
        print(f'Weights in the same directory {path}')
    elif (target_name == 'rlplays'):
        path = f'{target_name}_weights.bin'
        weights.tofile(path)

        # Write config to resources/<env_name>/<env_name>_config.ini
        # Contains the weights count+path and env args
        config_str = output_config_()
        with open(f'{target_name}_config.ini', 'w') as f:
            f.write(config_str)
        print(f'Saved {len(weights)} weights to {path} / config in {target_name}_config.ini')
    else:
        path = f'{args["env_name"]}_weights.bin'
        weights.tofile(path)
        print(f'Saved {len(weights)} weights to {path}')
    os._exit(0)

def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    env_name = args['env_name']
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])

def load_env(env_name, args):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])

def load_policy(args, vecenv, env_name=''):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    device = args['train']['device']
    policy_cls = getattr(env_module.torch, args['policy_name'])
    policy = policy_cls(vecenv.driver_env, **args['policy'])

    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['rnn'])

    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['neptune']:
            path = NeptuneLogger(args, load_id, mode='read-only').download()
        elif args['wandb']:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # for k in state_dict.keys():
        #     print(f"Loading weight {k} with shape {state_dict[k].shape}")
        # for k in policy.state_dict().keys():
        #     print(f"- Policy weight {k} with shape {policy.state_dict()[k].shape}")
        # NOTE: If there is an error here, it like'y the default policy params under [policy] do not match [rnn] params (e.g. hidden_size).
        #       Use the printouts above to debug.
        policy.load_state_dict(state_dict)
        #state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        #optim_state = torch.load(state_path)['optimizer_state_dict']
        #pufferl.optimizer.load_state_dict(optim_state)

    if load_path is not None:
      print(f'Loaded model from {load_path}')
    return policy

def load_config(env_name, parser=None):
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))
    config = process_config(p, parser=parser)
    pufferlib.PufferEnv.global_config = config
    return config

def load_config_file(file_path, fill_in_default=True, parser=None):
    if not os.path.exists(file_path):
        raise pufferlib.APIUsageError('No config file found')

    config_paths = [file_path]

    if fill_in_default:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
        # Process the puffer defaults first
        config_paths.insert(0, os.path.join(puffer_dir, 'config/default.ini'))

    p = configparser.ConfigParser()
    p.read(config_paths)

    return process_config(p, parser=parser)

def make_parser():
    '''Creates the argument parser with default PufferLib arguments.'''
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
    parser.add_argument('--neptune-name', type=str, default='pufferai')
    parser.add_argument('--neptune-project', type=str, default='ablations')
    parser.add_argument('--no-model-upload', action='store_true', help='Do not upload models to wandb or neptune')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    parser.add_argument('--profile.name', type=str, default='', help='Name for profiler trace using pufferl.py profile envs')
    parser.add_argument('--profile.eval', type=int, default=1, help='Whether to profile eval loop using pufferl.py profile envs')
    parser.add_argument('--profile.train', type=int, default=1, help='Whether to profile core train loop using pufferl.py profile envs')
    parser.add_argument('--profile.trace', type=int, default=0, help='Whether to export a CUDA trace (open the file using ui.perfetto.dev)')
    parser.add_argument('--profile.memory', type=int, default=0, help='Whether to enable CUDA memory profiling (Must have used PUFFER_SINGLE_THREADED=1 setup.py install)')
    return parser

def process_config(config, parser=None):
    if parser is None:
        parser = make_parser()

    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    def auto_type(value):
        """Type inference for numeric args that use 'auto' as a default value"""
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        return float(value)

    for section in config.sections():
        for key in config[section]:
            try:
                value = ast.literal_eval(config[section][key])
            except:
                value = config[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=value,
                type=auto_type if value == 'auto' else type(value)
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['env'] = args['env_name'] or ''  # for trainer dashboard
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = 'Usage: puffer [train, eval, sweep, autotune, profile, export] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == 'train':
        train(env_name=env_name)
    elif mode == 'eval':
        eval(env_name=env_name)
    elif mode == 'sweep':
        sweep(env_name=env_name)
    elif mode == 'autotune':
        autotune(env_name=env_name)
    elif mode == 'profile':
        profile(env_name=env_name)
    elif mode == 'export':
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()
