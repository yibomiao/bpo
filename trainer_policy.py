import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from transformers import logging

logging.set_verbosity_warning()
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
os.environ["WANDB_MODE"] = "offline"
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
import warnings
# warnings.filterwarnings("ignore")
# import bert_score
# from bert_score import BERTScorer

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
	"""Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
	Args:
		:attr:`A` (Tensor):
			The tensor to compute the Cholesky decomposition of
		:attr:`upper` (bool, optional):
			See torch.cholesky
		:attr:`out` (Tensor, optional):
			See torch.cholesky
		:attr:`jitter` (float, optional):
			The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
			as 1e-6 (float) or 1e-8 (double)
	"""
	try:
		L = torch.linalg.cholesky(A, upper=upper, out=out)
		return L
	except RuntimeError as e:
		isnan = torch.isnan(A)
		if isnan.any():
			raise ValueError(
				f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
			)

		if jitter is None:
			jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
		Aprime = A.clone()
		jitter_prev = 0
		for i in range(10):
			jitter_new = jitter * (10 ** i)
			Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
			jitter_prev = jitter_new
			try:
				L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
				warnings.warn(
					f"A not p.d., added jitter of {jitter_new} to the diagonal",
					RuntimeWarning,
				)
				return L
			except RuntimeError:
				continue
		raise e

class Additional_hps(torch.nn.Module):
    def __init__(self,config):
        super(Additional_hps, self).__init__()
        self.register_parameter(name="hp1", param=torch.nn.Parameter(torch.tensor([0.5])))  #[0.9741]
        # self.register_parameter(name="hp1", param=torch.nn.Parameter(torch.ones(1))) 
        self.register_parameter(name="hp2", param=torch.nn.Parameter(torch.tensor([0.5]))) 
        # self.register_parameter(name="hp2", param=torch.nn.Parameter(torch.ones(1))) 


def gp_sample_and_kl(policy_kernel_matrix, ref_kernel_matrix, predict):
    mean = torch.zeros(predict.shape[0]).cuda()
    # print("predict",predict)
    # print("kernel_matrix",kernel_matrix)
    # L_ref= psd_safe_cholesky(ref_kernel_matrix)
    # L_policy= psd_safe_cholesky(policy_kernel_matrix)
    # print('L',L)

    # predict_multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=predict, covariance_matrix=policy_kernel_matrix)
    predict_multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=policy_kernel_matrix)
    noise_multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=policy_kernel_matrix)
    multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=ref_kernel_matrix)
    # predict_multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=predict, scale_tril=L_policy)
    # multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, scale_tril=L_ref)
    # noise_multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, scale_tril=L_policy)

    # samples = multivariate_normal.sample(sample_shape=torch.Size([1]))  # 采样1个样本，可以根据需要更改数量
    samples = noise_multivariate_normal.rsample(sample_shape=torch.Size([256]))  # 采样1个样本，可以根据需要更改数量
    kl = torch.distributions.kl.kl_divergence(predict_multivariate_normal,multivariate_normal)
    # print("kl",kl)
    return samples, kl


#CUDA_VISIBLE_DEVICES="1,2" python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=16 batch_size=64 eval_batch_size=16 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/liymai24/sjtu/yibo/direct-preference-optimization-main/.cache/root/anthropic_dpo_pythia28_2023-12-27_08-16-22_907411/step-39936/policy.pt debug=true
def bayes_preference_loss(rank: int,
                    world_size: int,
                    batch: Dict[str, Union[List, torch.LongTensor]],
                    policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    reference_chosen_hidden: torch.FloatTensor,
                    reference_rejected_hidden: torch.FloatTensor,
                    policy_chosen_hidden: torch.FloatTensor,
                    policy_rejected_hidden: torch.FloatTensor,
                    all_policy_ps: torch.FloatTensor,
                    all_reference_ps: torch.FloatTensor,
                    # bert_scorer,
                    all_additional_hps,
                    train_process,
                    beta: float,
                    alpha: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

    batch_size = policy_chosen_logps.shape[0]
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    ref_kernel_matrix = torch.zeros(batch_size*2,batch_size*2).cuda()
    policy_kernel_matrix = torch.zeros(batch_size*2,batch_size*2).cuda()

    # 计算余弦相似度
    ref_similarity_1 = F.cosine_similarity(reference_chosen_hidden.unsqueeze(1), reference_chosen_hidden.unsqueeze(0), dim=2)
    ref_similarity_2 = F.cosine_similarity(reference_chosen_hidden.unsqueeze(1), reference_rejected_hidden.unsqueeze(0), dim=2)
    ref_similarity_3 = F.cosine_similarity(reference_rejected_hidden.unsqueeze(1), reference_chosen_hidden.unsqueeze(0), dim=2)
    ref_similarity_4 = F.cosine_similarity(reference_rejected_hidden.unsqueeze(1), reference_rejected_hidden.unsqueeze(0), dim=2)

    policy_similarity_1 = F.cosine_similarity(policy_chosen_hidden.unsqueeze(1), policy_chosen_hidden.unsqueeze(0), dim=2)
    policy_similarity_2 = F.cosine_similarity(policy_chosen_hidden.unsqueeze(1), policy_rejected_hidden.unsqueeze(0), dim=2)
    policy_similarity_3 = F.cosine_similarity(policy_rejected_hidden.unsqueeze(1), policy_chosen_hidden.unsqueeze(0), dim=2)
    policy_similarity_4 = F.cosine_similarity(policy_rejected_hidden.unsqueeze(1), policy_rejected_hidden.unsqueeze(0), dim=2)

    ref_kernel_matrix[:batch_size,:batch_size] = ref_similarity_1.reshape(batch_size,batch_size)
    ref_kernel_matrix[:batch_size,batch_size:] = ref_similarity_2.reshape(batch_size,batch_size)
    ref_kernel_matrix[batch_size:,:batch_size] = ref_similarity_3.reshape(batch_size,batch_size)
    ref_kernel_matrix[batch_size:,batch_size:] = ref_similarity_4.reshape(batch_size,batch_size)

    policy_kernel_matrix[:batch_size,:batch_size] = policy_similarity_1.reshape(batch_size,batch_size)
    policy_kernel_matrix[:batch_size,batch_size:] = policy_similarity_2.reshape(batch_size,batch_size)
    policy_kernel_matrix[batch_size:,:batch_size] = policy_similarity_3.reshape(batch_size,batch_size)
    policy_kernel_matrix[batch_size:,batch_size:] = policy_similarity_4.reshape(batch_size,batch_size)
    # batch_size = policy_chosen_logps.shape[0]
    # identity_matrix = torch.eye(2*batch_size).cuda()
    # kernel_matrix = kernel_matrix * (1 - identity_matrix) + identity_matrix

    g_pre = policy_chosen_logps - reference_chosen_logps
    g_rej = policy_rejected_logps - reference_rejected_logps

    # print("kernel_matrix",kernel_matrix)
    # print("g_pre",g_pre)
    # print("g_rej",g_rej)
    kl_divergence = F.kl_div(all_policy_ps.log(), all_reference_ps, reduction='sum')
    # print("kl_divergence shape",kl_divergence.shape)
    
    g = torch.cat((g_pre,g_rej),dim = 0) * beta
    # l2_norm = torch.norm(g, p=2)
    # print("g",g)
    policy_kernel_matrix = policy_kernel_matrix * torch.nn.functional.softplus(all_additional_hps.hp1) + all_additional_hps.hp2
    # print("all_additional_hps.hp1",all_additional_hps.hp1) #from cuda:0 -cuda:3 根据FSDP，最后统一更新？

    ref_kernel_matrix.diagonal(dim1=-2, dim2=-1).add_(1e-4)
    policy_kernel_matrix.diagonal(dim1=-2, dim2=-1).add_(1e-4)

    # print("ref_kernel_matrix",ref_kernel_matrix)
    # print("policy_kernel_matrix",policy_kernel_matrix)
    samples, kl = gp_sample_and_kl(policy_kernel_matrix, ref_kernel_matrix,g)


    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        sample_loss = samples[:,:batch_size] - samples[:,batch_size:]
        # print(rank)
        # print(world_size)
        sample_loss_all = all_gather_if_needed(sample_loss, rank, world_size)
        # print("sample_loss shape",sample_loss.shape) # 256* batchsize
        # print("sample_loss_all shape",sample_loss_all.shape) #(256*worldsize)* batchsize
        # print("sample_loss shape",sample_loss.shape) #sample_loss shape torch.Size([256, 8])  
        # print("logits shape",logits.shape) #logits shape torch.Size([8])
        # print("F.logsigmoid(beta * logits.unsqueeze(0) + sample_loss) shape",F.logsigmoid(beta * logits.unsqueeze(0) + sample_loss).shape) #shape torch.Size([256, 8])
        # print("F.logsigmoid(beta * logits.unsqueeze(0) + sample_loss).mean(0) shape",F.logsigmoid(beta * logits.unsqueeze(0) + sample_loss).mean(0).shape) #shape torch.Size([8])
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits.unsqueeze(0) + sample_loss_all).mean(0) * (1 - label_smoothing) - F.logsigmoid(-beta * logits.unsqueeze(0) + sample_loss_all).mean(0) * label_smoothing + (1/8*batch_size)* kl
        # losses = -F.logsigmoid(beta * logits + sample_loss) * (1 - label_smoothing) - F.logsigmoid(-beta * logits + sample_loss) * label_smoothing
        # losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits ) * label_smoothing + (1/8*batch_size)* kl
        # losses = -F.logsigmoid(beta * logits ) * (1 - label_smoothing) - F.logsigmoid(-beta * logits ) * label_smoothing - kl

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    if train_process:
        return losses, chosen_rewards, rejected_rewards, (1/8*batch_size)* kl, kl_divergence
    else:
        original_losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        return original_losses, chosen_rewards, rejected_rewards, (1/8*batch_size)* kl ,kl_divergence


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    # can refer to minidpo.py for more information
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        Choosen inputs index 0-batchsize/2, reject inputs index batchsize-batchsize
    """
    # print("batch['chosen_input_ids'] shape",batch['chosen_input_ids'].shape) #bs * sequence_len
    # print("batch['rejected_input_ids'] shape",batch['rejected_input_ids'].shape) #bs * sequence_len
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        # self.bert_scorer = BERTScorer(lang="en")
        self.all_additional_hps = Additional_hps(self.config).cuda()

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # print("self.tokenizer.pad_token_id",self.tokenizer.pad_token_id) #0
        
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model # in sft ref is none

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], return_hidden=False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        if return_hidden:
            concatenated_batch = concatenated_inputs(batch)
            # print("concatenated_batch",concatenated_batch)
            # print(concatenated_batch['concatenated_attention_mask'])
            # print("concatenated_batch['concatenated_attention_mask'] shape",concatenated_batch['concatenated_attention_mask'].shape)

            output = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'], output_hidden_states=True)
            all_logits = output.logits.to(torch.float32)   
            # print(type(all_hidden_states)) # turple
            # print(len(all_hidden_states)) # 33
            # print(all_hidden_states[-1].shape) # torch.Size([32, 512, 2560])

            # print("all_logits sahpe",all_logits.shape) #([32, 512, 50304]) bs,seqlen,vocubulary_size
            p_reference = all_logits.softmax(-1)
            all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
            # print("all_logps shape",all_logps.shape) # batchsize
            chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
            # print("batch['chosen_input_ids'].shape[0]",batch['chosen_input_ids'].shape[0]) # 16 batchsize/2
            rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
            all_hidden_states = output.hidden_states[-1].float()

            with torch.no_grad():
                weights_for_non_padding = concatenated_batch['concatenated_attention_mask'] * torch.arange(start=1, end=all_hidden_states.shape[1] + 1).cuda().unsqueeze(0)
                sum_embeddings = torch.sum(all_hidden_states * weights_for_non_padding.unsqueeze(-1), dim=1).float()
                # nonzero_count_per_row = torch.count_nonzero(weights_for_non_padding, dim=1).unsqueeze(-1)
                # print("Nonzero Count Per Row:", nonzero_count_per_row)
                num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).float().unsqueeze(-1)
                sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
                # print("sentence_embeddings",sentence_embeddings)
                # print("sentence_embeddings shape",sentence_embeddings.shape)
                chosen_hidden = sentence_embeddings[:batch['chosen_input_ids'].shape[0]] #batchsize * 2560
                rejected_hidden = sentence_embeddings[batch['rejected_input_ids'].shape[0]:]
            return chosen_logps, rejected_logps, chosen_hidden.detach().float(), rejected_hidden.detach().float(), p_reference
        else:
            concatenated_batch = concatenated_inputs(batch)
            # print("concatenated_batch",concatenated_batch)
            all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)    
            
            # print("all_logits sahpe",all_logits.shape) #([32, 512, 50304]) bs,seqlen,vocubulary_size
            all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
            # print("all_logps shape",all_logps.shape) # batchsize
            chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
            rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
            return chosen_logps, rejected_logps

    def concatenated_policy_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        concatenated_batch = concatenated_inputs(batch)
        output = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'], output_hidden_states=True)
        all_logits = output.logits.to(torch.float32)   

        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        p_policy = all_logits.softmax(-1)
        # print("all_logps shape",all_logps.shape) # batchsize
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        # print("batch['chosen_input_ids'].shape[0]",batch['chosen_input_ids'].shape[0]) # 16 batchsize/2
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        all_hidden_states = output.hidden_states[-1].float()

        weights_for_non_padding = concatenated_batch['concatenated_attention_mask'] * torch.arange(start=1, end=all_hidden_states.shape[1] + 1).cuda().unsqueeze(0)
        sum_embeddings = torch.sum(all_hidden_states * weights_for_non_padding.unsqueeze(-1), dim=1).float()
        # nonzero_count_per_row = torch.count_nonzero(weights_for_non_padding, dim=1).unsqueeze(-1)
        # print("Nonzero Count Per Row:", nonzero_count_per_row)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
        # print("sentence_embeddings",sentence_embeddings)
        # print("sentence_embeddings shape",sentence_embeddings.shape)
        chosen_hidden = sentence_embeddings[:batch['chosen_input_ids'].shape[0]].float() #batchsize * 2560
        rejected_hidden = sentence_embeddings[batch['rejected_input_ids'].shape[0]:].float()
        # print("chosen_hidden dtype",chosen_hidden.dtype)

        return chosen_logps, rejected_logps, chosen_hidden, rejected_hidden, p_policy


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'
        if loss_config.name in {'dpo', 'ipo'}:
            policy_chosen_logps, policy_rejected_logps, policy_chosen_hidden, policy_rejected_hidden, all_policy_ps = self.concatenated_policy_forward(self.policy, batch)

            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, reference_chosen_hidden, reference_rejected_hidden, all_reference_ps = self.concatenated_forward(self.reference_model, batch, return_hidden=True)
            if loss_config.name == 'dpo':
                loss_kwargs = {'beta': loss_config.beta, 'alpha': loss_config.alpha, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'alpha': loss_config.alpha, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')

            # print("batch['chosen_input_ids'] shape",batch['chosen_input_ids'].shape)
            # losses, chosen_rewards, rejected_rewards = preference_loss(
            #     policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
            if train_test == 'train':
                losses, chosen_rewards, rejected_rewards, kl,new_addkl = bayes_preference_loss(self.rank,self.world_size,
                    batch, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, reference_chosen_hidden, reference_rejected_hidden, policy_chosen_hidden, policy_rejected_hidden  ,all_policy_ps,all_reference_ps,self.all_additional_hps, train_process=True, **loss_kwargs)
            else:
                losses, chosen_rewards, rejected_rewards, kl,new_addkl = bayes_preference_loss(self.rank,self.world_size,
                    batch, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, reference_chosen_hidden, reference_rejected_hidden , policy_chosen_hidden, policy_rejected_hidden ,all_policy_ps,all_reference_ps,self.all_additional_hps, train_process=False, **loss_kwargs)
            # print("losses ",losses)
            # print("losses shape",losses.shape)
            # print("self.rank",self.rank)
            # print("self.world_size",self.world_size)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)
            kl = all_gather_if_needed(kl, self.rank, self.world_size)
            new_addkl = all_gather_if_needed(new_addkl, self.rank, self.world_size)
            # print("chosen_rewards shape",chosen_rewards.shape) #chosen_rewards shape torch.Size([32])-》eval

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/kl'] = kl.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/new_addkl'] = new_addkl.cpu().numpy().tolist()
            # metrics[f'kl_{train_test}/kl'] = kl.cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        # print("all_devices_losses shape",all_devices_losses.shape)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()
        # exit()
        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')

        param_groups = [
            {'params': self.policy.parameters(), 'lr': self.config.lr},
            {'params': [self.all_additional_hps.hp1, self.all_additional_hps.hp2], 'lr': 5e-4}
        ]
        self.optimizer = getattr(torch.optim, self.config.optimizer)(param_groups, lr=0)  # 这里的 lr=0 是一个占位符，实际的学习率由 param_groups 里指定

        # all_parameters = list(self.policy.parameters()) + [self.all_additional_hps.hp1, self.all_additional_hps.hp2] #hp1和hp2不需要存下来，因为最后只是想要policy模型
        # self.optimizer = getattr(torch.optim, self.config.optimizer)(all_parameters, lr=self.config.lr)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in tqdm.tqdm(self.train_iterator,desc='Training process'):

            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                torch.cuda.empty_cache()
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {'dpo', 'ipo'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    torch.cuda.empty_cache()
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {'dpo', 'ipo'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {'dpo', 'ipo'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {'dpo', 'ipo'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
            ### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items(): 
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                mean_train_metrics['hps/hp1'] = self.all_additional_hps.hp1.item()
                mean_train_metrics['hps/hp2'] = self.all_additional_hps.hp2.item()
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        # save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
        #     optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        # if self.rank == 0:
        #     self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        # del optimizer_state_dict
        # dist.barrier()

        # if self.rank == 0:
        #     scheduler_state_dict = self.scheduler.state_dict()
        #     self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        # dist.barrier()
        

class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        
        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        