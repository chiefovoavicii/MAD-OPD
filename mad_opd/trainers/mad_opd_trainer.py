# Copyright 2026 The MAD-OPD authors. Licensed under the Apache License, Version 2.0.
"""MAD-OPD trainer: On-Policy Distillation with Multi-Agent Debate supervision.

Implements the training loop of Paper Section 4 on top of TRL's ``GKDTrainer``:

* student samples an on-policy sequence for state s_m;
* K teachers debate for R rounds (Paper Eq. 5), yielding the privileged
  transcript H and teacher-reported confidence scores;
* confidences are turned into dynamic weights w_k via softmax (Paper Eq. 7);
* each teacher force-decodes the student's tokens conditioned on H; the
  student is supervised by the confidence-weighted per-teacher divergence
  (Paper Eq. 8);
* divergence D is JSD for agentic tasks and reverse-KL for code generation
  (Paper §3.3, Remark 1).
"""
import inspect
import os
import re
import random
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, Tuple, Union, List
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import trl
from accelerate.utils import gather_object, is_peft_model
from packaging import version
from transformers import PreTrainedModel
from trl import GKDTrainer as HFGKDTrainer
from trl import SFTTrainer as HFSFTTrainer

from swift.template import TemplateInputs
from swift.trainers import SwiftMixin, disable_gradient_checkpointing
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response, to_device,
                         unwrap_model_for_generation)
# ``rollout_mixin`` / ``utils`` live in the upstream ``swift.rlhf_trainers``
# package — reuse verbatim instead of shipping duplicates.
from swift.rlhf_trainers.rollout_mixin import DataType, RolloutTrainerMixin
from swift.rlhf_trainers.utils import (get_gather_if_zero3_context, identity_data_collator,
                                       patch_profiling_context, patch_profiling_decorator,
                                       prepare_deepspeed)
from .mad_opd_core import (compute_mad_distillation_loss, multi_teacher_jsd_loss,
                           generalized_jsd_loss_single)
from .vllm_teacher_manager import VLLMTeacherManager

# Optional Liger fused JSD kernel (used on the single-teacher path).
try:
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    _liger_kernel_available = True
except ImportError:
    _liger_kernel_available = False

# Reset HF parent ``__init__`` so ``SwiftMixin`` can drive initialisation.
# The attribute is not always declared directly on the TRL class (TRL inherits
# from SFTTrainer, which inherits from ``transformers.Trainer``); use a guarded
# ``delattr`` so the import does not crash on newer TRL versions.
for _parent in (HFGKDTrainer, HFSFTTrainer):
    if '__init__' in _parent.__dict__:
        delattr(_parent, '__init__')

logger = get_logger()
if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab


class DataSource(str, Enum):
    """Source of the response sequence consumed by the trainer.

    - STUDENT: on-policy mode, response sampled from the student model
    - TEACHER: sequential-KD mode, response sampled from the teacher model
    - DATASET: off-policy mode, response read from the dataset
    """
    STUDENT = 'student'
    TEACHER = 'teacher'
    DATASET = 'dataset'

class ToolACEGroupedBatchSampler:

    def __init__(
        self,
        group_ids: list,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        data_seed: Optional[int],
        tp_size: int = 1,
    ):
        self.tp_size = tp_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed

        group_to_indices = defaultdict(list)
        for idx, group_id in enumerate(group_ids):
            if group_id is None:
                group_to_indices[f'__singleton_{idx}'] = [idx]
            else:
                group_to_indices[group_id].append(idx)

        self.groups = list(group_to_indices.values())
        self.total_samples = len(group_ids)

    @property
    def rank(self):
        return (dist.get_rank() // self.tp_size) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return (dist.get_world_size() // self.tp_size) if dist.is_initialized() else 1

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            group_order = torch.randperm(len(self.groups), generator=generator).tolist()
        else:
            group_order = list(range(len(self.groups)))

        all_indices = []
        for group_idx in group_order:
            all_indices.extend(self.groups[group_idx])

        shard_indices = all_indices[self.rank::self.world_size]

        batch = []
        for idx in shard_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch

    def set_epoch(self, epoch: int):
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        shard_size = self.total_samples // self.world_size
        if self.drop_last:
            return shard_size // self.batch_size
        else:
            return (shard_size + self.batch_size - 1) // self.batch_size


class MADOPDTrainer(RolloutTrainerMixin, SwiftMixin, HFGKDTrainer):
    """MAD-OPD trainer (extends TRL's ``GKDTrainer``)."""

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_model = kwargs.pop('teacher_model', None)
        teacher_model_2 = kwargs.pop('teacher_model_2', None)
        teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        self.vllm_client = kwargs.pop('vllm_client', None)
        use_vllm_teachers_flag = kwargs.pop('use_vllm_teachers', False)
        # Pop kwargs that ms-swift's SwiftRLHF pipeline adds but the stock
        # ``transformers.Trainer.__init__`` does not accept.  We just store
        # them on ``self`` and never forward them up the chain.
        self.gkd_logits_topk = kwargs.pop('gkd_logits_topk', None)
        self.teacher_model_server = kwargs.pop('teacher_model_server', None)
        self._teacher_use_disable_adapter = kwargs.pop('teacher_use_disable_adapter', False)
        super().__init__(model, None, *_args, **kwargs)

        args = kwargs['args']
        self.lmbda = args.lmbda
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.generation_config = model.generation_config
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0

        self.gkd_algorithm = args.gkd_algorithm
        self.teacher_weights = args.teacher_weights or [0.5, 0.5]
        self.mad_max_debate_tokens = args.mad_max_debate_tokens
        self.mad_debate_temperature = args.mad_debate_temperature
        self.mad_debate_top_p = args.mad_debate_top_p
        self.mad_debate_top_k = args.mad_debate_top_k
        self.mad_t1_max_tokens = args.mad_t1_max_tokens
        self.mad_t2_max_tokens = args.mad_t2_max_tokens
        self.mad_dynamic_weights = getattr(args, 'mad_dynamic_weights', False)
        self.mad_debate_rounds = getattr(args, 'mad_debate_rounds', 2)
        self.jsd_top_k = getattr(args, 'jsd_top_k', None)
        if self.jsd_top_k is not None:
            logger.info(
                f'JSD top-k truncation enabled: top_k={self.jsd_top_k}. '
                f'JSD loss will only consider the top-{self.jsd_top_k} teacher tokens per position.'
            )

        self.teacher_data_split = getattr(args, 'teacher_data_split', 0.7)

        # Step-level supervision configuration

        self._prepare_logging()

        self._prepare_liger_loss()

        self.teacher_ds3_gather_for_generation = args.ds3_gather_for_generation
        self.is_teacher_ds3 = None

        # default algorithm uses standard teacher forward pass (no context generation),
        # so vLLM teacher mode is not needed and would cause errors (teacher_model becomes a string path)
        # multi_teacher supports vLLM teacher mode: uses sidecar force decode for both teachers
        if use_vllm_teachers_flag and self.gkd_algorithm in ('opd',):
            logger.info(
                f'[GKD] Algorithm "{self.gkd_algorithm}" does not require vLLM teacher interaction. '
                f'Automatically disabling use_vllm_teachers.'
            )
            use_vllm_teachers_flag = False

        self.use_vllm_teachers = use_vllm_teachers_flag
        self.vllm_teacher_manager = None

        if self.use_vllm_teachers:

            teacher_1_url = getattr(args, 'vllm_teacher_1_url', 'http://localhost:8001/v1')
            teacher_2_url = getattr(args, 'vllm_teacher_2_url', None)
            vllm_max_len_default = getattr(args, 'vllm_teacher_max_model_len', 16384)
            vllm_max_len_1 = getattr(args, 'vllm_teacher_1_max_model_len', None) or vllm_max_len_default
            vllm_max_len_2 = getattr(args, 'vllm_teacher_2_max_model_len', None) or vllm_max_len_default

            # Force Decode sidecar URLs for full-vocabulary logits
            sidecar_1_url = getattr(args, 'vllm_sidecar_1_url', None)
            sidecar_2_url = getattr(args, 'vllm_sidecar_2_url', None)

            sidecar_max_length = getattr(args, 'vllm_sidecar_max_length', 40960)
            teacher_timeout = getattr(args, 'vllm_teacher_timeout', 600.0)

            self.vllm_teacher_manager = VLLMTeacherManager(
                teacher_1_url=teacher_1_url,
                teacher_2_url=teacher_2_url,
                tokenizer=self.processing_class,
                max_model_len_1=vllm_max_len_1,
                max_model_len_2=vllm_max_len_2,
                timeout=teacher_timeout,
                max_retries=3,
                sidecar_1_url=sidecar_1_url,
                sidecar_2_url=sidecar_2_url,
                sidecar_max_length=sidecar_max_length,
                sidecar_top_k=self.jsd_top_k,
            )

            self.teacher_model = teacher_model
            self.teacher_model_2 = teacher_model_2
            self.is_teacher_ds3 = False
            self.teacher_ds3_gather_for_generation = False

            if teacher_model is not None and hasattr(teacher_model, 'cpu'):
                del teacher_model
            if teacher_model_2 is not None and hasattr(teacher_model_2, 'cpu'):
                del teacher_model_2
            torch.cuda.empty_cache()

            logger.info(
                f'[vLLM Teachers] Initialized HTTP server mode: '
                f'teacher_1={teacher_1_url}, teacher_2={teacher_2_url}'
            )
        elif teacher_model is None:
            self.teacher_model = None
            self.teacher_model_2 = None
            self.is_teacher_ds3 = False
            self.teacher_ds3_gather_for_generation = False
            logger.info('[GKD] No teacher model provided, skipping teacher initialization')
        else:
            if self.is_deepspeed_enabled:
                if teacher_deepspeed_config is not None:
                    self.is_teacher_ds3 = teacher_deepspeed_config.get('zero_optimization', {}).get('stage') == 3
                    if not self.is_teacher_ds3:
                        self.teacher_ds3_gather_for_generation = False
                    self.teacher_model = prepare_deepspeed(
                        teacher_model, self.accelerator, deepspeed_config=teacher_deepspeed_config, training_args=args)
                else:
                    self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            elif self.is_fsdp_enabled:
                from swift.rlhf_trainers.utils import prepare_fsdp
                self.teacher_model = prepare_fsdp(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

            self.teacher_model.eval()

            if self.args.offload_teacher_model:
                self.offload_model(self.accelerator.unwrap_model(self.teacher_model))

            # Initialise the second teacher model used by MAD-OPD.
            self.teacher_model_2 = None
            if teacher_model_2 is not None and self.gkd_algorithm in ('mad_opd', 'mt_opd'):
                if self.is_deepspeed_enabled:
                    if teacher_deepspeed_config is not None:
                        self.teacher_model_2 = prepare_deepspeed(
                            teacher_model_2, self.accelerator, deepspeed_config=teacher_deepspeed_config, training_args=args)
                    else:
                        self.teacher_model_2 = prepare_deepspeed(teacher_model_2, self.accelerator)
                elif self.is_fsdp_enabled:
                    from swift.rlhf_trainers.utils import prepare_fsdp
                    self.teacher_model_2 = prepare_fsdp(teacher_model_2, self.accelerator)
                else:
                    self.teacher_model_2 = self.accelerator.prepare_model(teacher_model_2, evaluation_mode=True)

                self.teacher_model_2.eval()

                if self.args.offload_teacher_model:
                    self.offload_model(self.accelerator.unwrap_model(self.teacher_model_2))

                logger.info(f'Initialized second teacher model for {self.gkd_algorithm} algorithm')


        # `code_mode` switches the MAD debate prompts to their code-generation
        # variants (Paper §3.3 — reverse-KL regime).  When False we use the
        # agentic / math-reasoning prompts by default.
        self.code_mode = getattr(args, 'code_mode', False)
        if self.code_mode:
            logger.info(
                '[Code Mode] Enabled code task distillation mode: '
                'using code-specific debate prompts, GT injection disabled in Force Decode context'
            )

        self.toolace_mode = args.toolace_mode
        self.toolace_onpolicy_mode = args.toolace_onpolicy_mode
        self.toolace_max_turns = args.toolace_max_turns
        self.toolace_mask_tool_response = args.toolace_mask_tool_response

        if self.code_mode and self.toolace_mode:
            raise ValueError(
                '`--code_mode` and `--toolace_mode` are mutually exclusive: '
                'choose code-generation training (code_mode=true) or '
                'agentic ToolACE training (toolace_mode=true), not both.'
            )

        if self.toolace_mode:
            if self.toolace_onpolicy_mode == 'full':
                from data.toolace_env import ToolACEMultiTurnManager
                self.toolace_manager = ToolACEMultiTurnManager(max_turns=self.toolace_max_turns)
                logger.info(
                    f'Initialized ToolACE multi-turn distillation mode (full on-policy) '
                    f'(max_turns={self.toolace_max_turns}, mask_tool_response={self.toolace_mask_tool_response})'
                )
            else:
                self.toolace_manager = None
                logger.info(
                    f'Initialized ToolACE multi-turn distillation mode (teacher_forcing) '
                    f'(mask_tool_response={self.toolace_mask_tool_response})'
                )

        self.prepare_rollout()

        self.maybe_activation_offload_context = nullcontext()

        self._trl_version_gte_0_24 = version.parse(trl.__version__) >= version.parse('0.24')

        if self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()

    def _get_data_collator(self, args, template):
        return identity_data_collator

    def get_train_dataloader(self, skip_batches=0):
        if not getattr(self, 'toolace_mode', False):
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        if train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        has_group_id = '_toolace_group_id' in train_dataset.column_names if hasattr(train_dataset, 'column_names') else False
        if not has_group_id:
            logger.warning(
                '[ToolACE] Dataset does not contain _toolace_group_id column. '
                'Falling back to default dataloader without grouped sampling.'
            )
            return super().get_train_dataloader()

        from functools import partial
        from swift.dataloader import DataLoaderShard

        args = self.args
        group_ids = train_dataset['_toolace_group_id']

        batch_sampler = ToolACEGroupedBatchSampler(
            group_ids=group_ids,
            batch_size=self._train_batch_size,
            shuffle=args.train_dataloader_shuffle,
            drop_last=args.dataloader_drop_last,
            data_seed=args.data_seed,
            tp_size=(
                args.deepspeed['tensor_parallel']['autotp_size']
                if args.deepspeed and 'tensor_parallel' in args.deepspeed else 1
            ),
        )

        dataloader_params = {
            'collate_fn': self.data_collator,
            'num_workers': args.dataloader_num_workers,
            'pin_memory': args.dataloader_pin_memory,
            'persistent_workers': args.dataloader_persistent_workers,
            'prefetch_factor': args.dataloader_prefetch_factor,
        }

        def seed_worker(worker_id, num_workers=1, rank=0):
            import numpy as np
            worker_seed = (rank * num_workers + worker_id) + args.data_seed if args.data_seed else worker_id
            np.random.seed(worker_seed % (2**32))
            random.seed(worker_seed)

        dataloader_params['worker_init_fn'] = partial(
            seed_worker, num_workers=args.dataloader_num_workers, rank=args.process_index)

        if skip_batches > 0:
            from accelerate.data_loader import SkipBatchSampler
            batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_batches)

        dataloader_params['batch_sampler'] = batch_sampler
        dataloader = DataLoaderShard(train_dataset, device=self.accelerator.device, **dataloader_params)

        logger.info(
            f'[ToolACE] Using ToolACEGroupedBatchSampler with {len(batch_sampler)} batches, '
            f'{len(set(group_ids))} groups from {len(group_ids)} samples'
        )
        return dataloader

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        assert not self.template.padding_free, 'generate not support padding_free/packing.'

        prompt_input_ids = inputs['input_ids']

        model_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        model_inputs.pop('position_ids', None)
        model_inputs.pop('text_position_ids', None)

        kwargs = {}
        base_model = self.template.get_base_model(model)
        parameters = inspect.signature(base_model.generate).parameters
        if 'use_model_defaults' in parameters:
            kwargs['use_model_defaults'] = False

        with self.template.generate_context():
            if self.model.model_meta.is_multimodal:
                _, model_inputs = self.template.pre_forward_hook(model, None, model_inputs)

            generated_outputs = model.generate(
                **model_inputs, generation_config=generation_config, return_dict_in_generate=True, **kwargs)

        generated_tokens = generated_outputs.sequences

        if not self.template.skip_prompt:
            generated_tokens = torch.concat([prompt_input_ids, generated_tokens], dim=1)

        new_attention_mask = torch.ones_like(generated_tokens)

        new_labels = generated_tokens.clone()
        new_labels[:, :prompt_input_ids.shape[1]] = -100

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        new_position_ids = new_attention_mask.cumsum(dim=1) - 1
        new_position_ids[new_position_ids < 0] = 0
        inputs['position_ids'] = new_position_ids

        return generated_tokens, new_attention_mask, new_labels

    @patch_profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        data_source = inputs.pop('_data_source', DataSource.DATASET)

        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}

        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(inputs)
            model_inputs['logits_to_keep'] = inputs['logits_to_keep']

        if self.use_liger_gkd_loss:

            unwrapped_student = self.accelerator.unwrap_model(model)
            if is_peft_model(unwrapped_student):
                unwrapped_student = unwrapped_student.base_model.model
            base_student = getattr(unwrapped_student, getattr(unwrapped_student, 'base_model_prefix', 'model'),
                                   unwrapped_student)

            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            base_teacher = getattr(unwrapped_teacher, getattr(unwrapped_teacher, 'base_model_prefix', 'model'),
                                   unwrapped_teacher)

            student_outputs = base_student(**model_inputs, use_cache=False)

            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with load_context:
                with torch.no_grad(), disable_gradient_checkpointing(self.teacher_model,
                                                                     self.args.gradient_checkpointing_kwargs):
                    teacher_outputs = base_teacher(**model_inputs, use_cache=False)

                student_hidden = student_outputs.last_hidden_state[:, :-1]
                teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

                del student_outputs, teacher_outputs

                labels_mask = inputs['labels'] != -100
                masked_input_ids = torch.where(labels_mask, inputs['input_ids'],
                                               torch.full_like(inputs['input_ids'], -100))
                true_labels = masked_input_ids[:, 1:].contiguous()

                del labels_mask, masked_input_ids

                student_head = unwrapped_student.get_output_embeddings()
                teacher_head = unwrapped_teacher.get_output_embeddings()

                teacher_context = get_gather_if_zero3_context(self, is_zero3=self.is_teacher_ds3)(teacher_head.weight)
                student_context = get_gather_if_zero3_context(self)(student_head.weight)

                with teacher_context, student_context:
                    loss = self.liger_jsd_loss(
                        student_input=student_hidden,
                        student_weight=student_head.weight,
                        teacher_input=teacher_hidden,
                        teacher_weight=teacher_head.weight,
                        true_labels=true_labels,
                        student_bias=getattr(student_head, 'bias', None),
                        teacher_bias=getattr(teacher_head, 'bias', None),
                    )
                    loss /= student_hidden.shape[1]

                del student_hidden, teacher_hidden, true_labels
        else:

            # Skip the teacher forward pass when we already have per-teacher interaction results from the MAD debate round (logits come from force-decode).
            has_teacher_interaction = False
            if self.gkd_algorithm in ('mad_opd', 'mt_opd'):
                has_teacher_interaction = (hasattr(self, '_current_vllm_logits_ready')
                                           and self._current_vllm_logits_ready)
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']

            outputs_student = model(**model_inputs)

            shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
            mask = shifted_labels != -100

            # Skip the teacher forward pass when MAD interaction results are already available.
            if True:
                model_inputs.pop('labels', None)

            if has_teacher_interaction:
                shifted_student_logits = outputs_student.logits
                shifted_teacher_logits = None

                if hasattr(self, '_current_prompt_lengths') and self._current_prompt_lengths is not None:
                    self._current_prompt_lengths_for_loss = self._current_prompt_lengths
                else:
                    self._current_prompt_lengths_for_loss = []
                    for i in range(inputs['labels'].shape[0]):
                        prompt_len = (inputs['labels'][i] == -100).sum().item()
                        self._current_prompt_lengths_for_loss.append(prompt_len)

            else:
                # Only create load_context here to avoid unwrapping teacher_model when it's a string (vLLM mode)
                # load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
                # with torch.no_grad(), load_context, disable_gradient_checkpointing(
                #         self.teacher_model, self.args.gradient_checkpointing_kwargs):
                #     outputs_teacher = self.teacher_model(**model_inputs)
                labels_shifted = inputs['labels'][:, 1:]
                valid_mask = labels_shifted != -100

                valid_indices = valid_mask.nonzero(as_tuple=False)
                if valid_indices.numel() == 0:
                    loss = (outputs_student.logits * 0).sum()
                    if return_outputs:
                        return (loss, outputs_student)
                    return loss

                first_valid = valid_indices[:, 1].min().item()
                last_valid = valid_indices[:, 1].max().item() + 1

                shifted_student_logits = outputs_student.logits[:, first_valid:last_valid, :]
                shifted_labels = labels_shifted[:, first_valid:last_valid]

                load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
                with torch.no_grad(), load_context, disable_gradient_checkpointing(
                        self.teacher_model, self.args.gradient_checkpointing_kwargs):
                    outputs_teacher = self.teacher_model(**model_inputs)

                shifted_teacher_logits = outputs_teacher.logits[:, first_valid:last_valid, :].detach()
                

                stu_dim = shifted_student_logits.shape[-1]
                tea_dim = shifted_teacher_logits.shape[-1]
                if stu_dim != tea_dim:
                    min_dim = min(stu_dim, tea_dim)
                    shifted_student_logits = shifted_student_logits[..., :min_dim]
                    shifted_teacher_logits = shifted_teacher_logits[..., :min_dim]

            # Dispatch to the appropriate loss computation path.
            if self.gkd_algorithm == 'opd' or self.teacher_model_2 is None:
                loss = self.generalized_jsd_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=None,
                    beta=self.beta,
                    temperature=self.temperature,
                )

            elif self.gkd_algorithm in ('mad_opd', 'mt_opd'):

                has_vllm_logits = (hasattr(self, '_current_vllm_logits_ready')
                                   and self._current_vllm_logits_ready)

                if has_vllm_logits:
                    batch_results = self._current_batch_vllm_results
                    self._current_batch_vllm_results = None
                    self._current_vllm_logits_ready = False

                    prompt_lengths = getattr(self, '_current_prompt_lengths_for_loss', None)


                    logits_already_truncated = (
                        'logits_to_keep' in inputs
                        or (hasattr(self, 'get_use_logits_to_keep') and self.get_use_logits_to_keep(True)
                            and shifted_student_logits.shape[1] < inputs['input_ids'].shape[1])
                    )

                    valid_losses = []
                    all_mad_metrics = defaultdict(list)
                    
                    for i, result in enumerate(batch_results):
                        # Skip samples where both teacher logits are None (Force Decode OOM/failure)
                        if result['teacher_a_logits'] is None and result['teacher_b_logits'] is None:
                            logger.warning(f'[GKD] Skipping sample {i}: both teacher logits are None (force decode failed)')
                            continue

                        if logits_already_truncated:
                            sample_student_logits = shifted_student_logits[i:i+1, :, :]
                        else:
                            if prompt_lengths is not None and i < len(prompt_lengths):
                                prompt_len = prompt_lengths[i]
                            else:
                                sample_labels = shifted_labels[i]
                                valid_positions = (sample_labels != -100).nonzero(as_tuple=False)
                                if valid_positions.numel() == 0:
                                    continue
                                prompt_len = valid_positions[0].item()

                            sample_student_logits = shifted_student_logits[i:i+1, prompt_len - 1:-1, :]

                        if sample_student_logits.numel() == 0:
                            effective_prompt_len = 'N/A(truncated)' if logits_already_truncated else prompt_len
                            logger.warning(
                                f'[GKD] Skipping sample {i}: empty student logits '
                                f'(prompt_len={effective_prompt_len}, seq_len={shifted_student_logits.shape[1]})'
                            )
                            continue

                        if self.accelerator.is_main_process and self.state.global_step % 5 == 0:
                            ta_logits = result['teacher_a_logits']
                            tb_logits = result['teacher_b_logits']
                            ta_len = ta_logits.shape[1] if ta_logits is not None else 0
                            tb_len = tb_logits.shape[1] if tb_logits is not None else 0
                            stu_len = sample_student_logits.shape[1]

                            diag_prompt_len = 'N/A(truncated)' if logits_already_truncated else prompt_len
                            logger.info(
                                f'[MAD2-DIAG] Step {self.state.global_step} Sample {i} | '
                                f'prompt_len={diag_prompt_len}, full_seq_len={shifted_student_logits.shape[1]}, '
                                f'logits_truncated={logits_already_truncated}, '
                                f'student_response_len={stu_len}, '
                                f'teacher_a_len={ta_len}, teacher_b_len={tb_len}, '
                                f'len_diff_a={abs(stu_len - ta_len)}, len_diff_b={abs(stu_len - tb_len)}'
                            )

                            stu_mean = sample_student_logits.float().mean().item()
                            stu_std = sample_student_logits.float().std().item()
                            stu_max = sample_student_logits.float().max().item()
                            logger.info(
                                f'[MAD2-DIAG] Student logits stats: mean={stu_mean:.4f}, std={stu_std:.4f}, max={stu_max:.4f}'
                            )
                            if ta_logits is not None:
                                ta_mean = ta_logits.float().mean().item()
                                ta_std = ta_logits.float().std().item()
                                logger.info(f'[MAD2-DIAG] Teacher-A logits stats: mean={ta_mean:.4f}, std={ta_std:.4f}')
                            if tb_logits is not None:
                                tb_mean = tb_logits.float().mean().item()
                                tb_std = tb_logits.float().std().item()
                                logger.info(f'[MAD2-DIAG] Teacher-B logits stats: mean={tb_mean:.4f}, std={tb_std:.4f}')

                            effective_len = min(stu_len, ta_len, tb_len) if ta_logits is not None and tb_logits is not None else min(stu_len, max(ta_len, tb_len))
                            if effective_len > 0:
                                stu_top1 = sample_student_logits[0, :effective_len, :].argmax(dim=-1)
                                if ta_logits is not None:
                                    ta_top1 = ta_logits[0, :effective_len, :].argmax(dim=-1)
                                    agree_a = (stu_top1 == ta_top1).float().mean().item()
                                    logger.info(f'[MAD2-DIAG] Student-TeacherA top1 agreement: {agree_a:.2%}')
                                if tb_logits is not None:
                                    tb_top1 = tb_logits[0, :effective_len, :].argmax(dim=-1)
                                    agree_b = (stu_top1 == tb_top1).float().mean().item()
                                    logger.info(f'[MAD2-DIAG] Student-TeacherB top1 agreement: {agree_b:.2%}')

                            check_len = min(5, stu_len)
                            stu_first5 = sample_student_logits[0, :check_len, :].argmax(dim=-1).tolist()
                            stu_first5_text = self.processing_class.decode(stu_first5)
                            logger.info(f'[MAD2-DIAG] Student first 5 predicted tokens: {stu_first5} -> "{stu_first5_text}"')
                            if ta_logits is not None and ta_len >= check_len:
                                ta_first5 = ta_logits[0, :check_len, :].argmax(dim=-1).tolist()
                                ta_first5_text = self.processing_class.decode(ta_first5)
                                logger.info(f'[MAD2-DIAG] Teacher-A first 5 predicted tokens: {ta_first5} -> "{ta_first5_text}"')

                        sample_loss, sample_metrics = compute_mad_distillation_loss(
                            student_logits=sample_student_logits,
                            teacher_a_logits=result['teacher_a_logits'],
                            teacher_b_logits=result['teacher_b_logits'],
                            alpha=result['alpha'],
                            beta=result['beta'],
                            labels=None,
                            temperature=self.temperature,
                            jsd_beta=self.beta,
                            response_mask=None,
                            jsd_top_k=self.jsd_top_k,
                        )
                        if self.accelerator.is_main_process and self.state.global_step % 5 == 0:
                            logger.info(
                                f'[MAD2-DIAG] Sample {i} loss={sample_loss.item():.6f}, '
                                f'requires_grad={sample_loss.requires_grad}, '
                                f'alpha={result["alpha"]:.3f}, beta={result["beta"]:.3f}, '
                                f'metrics={sample_metrics}'
                            )

                        # Only keep losses with meaningful gradient signal
                        if sample_loss.requires_grad and sample_loss.item() > 1e-8:
                            valid_losses.append(sample_loss)
                        else:
                            logger.warning(
                                f'[GKD] Skipping sample {i}: loss={sample_loss.item():.2e}, '
                                f'requires_grad={sample_loss.requires_grad}'
                            )
                        for key, value in sample_metrics.items():
                            if isinstance(value, (int, float)):
                                all_mad_metrics[key].append(value)
                    if len(valid_losses) > 0:
                        # Changed: stack and mean only valid losses
                        loss = torch.stack(valid_losses).mean()
                    else:
                        loss = (shifted_student_logits * 0).sum()
                        logger.warning(
                            '[GKD] All samples skipped in MAD vLLM mode (all repetitive/failed). '
                            'Using zero-gradient loss to skip this micro-batch.'
                        )
                        all_mad_metrics['mad_all_samples_skipped'].append(1.0)

                    for key, values in all_mad_metrics.items():
                        if values:
                            self._metrics['train'][key].append(sum(values) / len(values))

            else:
                raise ValueError(f"Unknown gkd_algorithm: {self.gkd_algorithm}. "
                               f"Supported values: 'default', 'mad_opd'")

            # `shifted_teacher_logits` is None when interaction-based logits are already prepared; skip logging.
            if self.accelerator.is_main_process and hasattr(self, 'state') \
                    and self.state.global_step % 10 == 0 \
                    and shifted_teacher_logits is not None:
                self._debug_log_logits_diff(
                    shifted_student_logits, shifted_teacher_logits, loss,
                    shifted_teacher_2_logits=locals().get('shifted_teacher_2_logits'),
                )

            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss

        if return_outputs:
            if self.use_liger_gkd_loss:
                outputs_student = None
            return (loss, outputs_student)
        else:
            return loss
    # =========================================================================

    def _debug_log_student_output(self, encoded_inputs, data_source, raw_inputs=None):
        try:
            input_ids = encoded_inputs.get('input_ids')
            labels = encoded_inputs.get('labels')
            if input_ids is None or labels is None:
                return

            sample_ids = input_ids[0]
            sample_labels = labels[0]

            response_mask = sample_labels != -100
            response_token_count = response_mask.sum().item()
            total_token_count = sample_ids.shape[0]

            if response_token_count > 0:
                response_ids = sample_ids[response_mask]
                decode_ids = response_ids[-100:]
                response_preview = self.processing_class.decode(decode_ids, skip_special_tokens=True)
                if len(response_preview) > 200:
                    response_preview = '...' + response_preview[-200:]
            else:
                response_preview = '<empty>'

            batch_info = ''
            if raw_inputs is not None:
                batch_size = input_ids.shape[0]
                zero_count = sum(1 for b in range(batch_size) if (labels[b] != -100).sum().item() == 0)
                batch_info = f'batch_zero_response={zero_count}/{batch_size} | '

            logger.info(
                f'[GKD-DEBUG] Step {self.state.global_step} | '
                f'source={data_source.value} | algo={self.gkd_algorithm} | '
                f'total_tokens={total_token_count} | response_tokens={response_token_count} | '
                f'{batch_info}'
                f'student_output: {response_preview}'
            )

            if raw_inputs is not None and response_token_count == 0:
                tail_ids = sample_ids[-200:]
                tail_text = self.processing_class.decode(tail_ids, skip_special_tokens=False)
                logger.info(f'[GKD-DEBUG-DIAG] tail_200_tokens: {tail_text[-500:]}')

                if len(raw_inputs) > 0:
                    sample = raw_inputs[0]
                    msgs = sample.get('messages', [])
                    roles = [m.get('role', '?') for m in msgs]
                    contents_preview = []
                    for m in msgs:
                        c = m.get('content', '')
                        if c and isinstance(c, str):
                            contents_preview.append(f"{m.get('role','?')}({len(c)}chars): {c[:80]}...")
                        else:
                            contents_preview.append(f"{m.get('role','?')}: content={type(c).__name__}")
                    logger.info(f'[GKD-DEBUG-DIAG] num_messages={len(msgs)} | roles={roles}')
                    for cp in contents_preview[:6]:
                        logger.info(f'[GKD-DEBUG-DIAG]   {cp}')
        except Exception as exc:
            logger.debug(f'[GKD-DEBUG] Failed to log student output: {exc}')

    def _debug_log_logits_diff(self, student_logits, teacher_logits, loss,
                               shifted_teacher_2_logits=None):
        try:
            with torch.no_grad():
                num_positions = min(student_logits.shape[1], 20)
                stu_slice = student_logits[0, :num_positions].detach().float()
                tea_slice = teacher_logits[0, :num_positions].detach().float()

                stu_log_probs = F.log_softmax(stu_slice, dim=-1)
                tea_log_probs = F.log_softmax(tea_slice, dim=-1)

                tea_top_probs, tea_top_ids = tea_log_probs.max(dim=-1)
                stu_at_tea_top = stu_log_probs.gather(1, tea_top_ids.unsqueeze(-1)).squeeze(-1)

                prob_gap = (tea_top_probs - stu_at_tea_top).mean().item()

                stu_top_ids = stu_log_probs.argmax(dim=-1)
                agreement_rate = (stu_top_ids == tea_top_ids).float().mean().item()

                log_msg = (
                    f'[GKD-DEBUG] Step {self.state.global_step} | '
                    f'algo={self.gkd_algorithm} | loss={loss.item():.4f} | '
                    f'top1_agreement={agreement_rate:.2%} | '
                    f'mean_logprob_gap={prob_gap:.4f}'
                )

                if shifted_teacher_2_logits is not None:
                    tea2_slice = shifted_teacher_2_logits[0, :num_positions].detach().float()
                    tea2_log_probs = F.log_softmax(tea2_slice, dim=-1)
                    tea2_top_probs, tea2_top_ids = tea2_log_probs.max(dim=-1)
                    stu_at_tea2_top = stu_log_probs.gather(1, tea2_top_ids.unsqueeze(-1)).squeeze(-1)
                    prob_gap_2 = (tea2_top_probs - stu_at_tea2_top).mean().item()
                    agreement_2 = (stu_top_ids == tea2_top_ids).float().mean().item()
                    log_msg += (
                        f' | teacher2_top1_agreement={agreement_2:.2%}'
                        f' | teacher2_logprob_gap={prob_gap_2:.4f}'
                    )

                logger.info(log_msg)
        except Exception as exc:
            logger.debug(f'[GKD-DEBUG] Failed to log logits diff: {exc}')

    def _prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False) -> Dict[str, torch.Tensor]:
        from swift.rlhf_trainers.utils import replace_assistant_response_with_ids

        template = self.template
        batch_encoded_inputs = []

        mode = 'transformers' if encode_prompt_only else 'train'

        _is_seqkd = False
        _diag_step = getattr(self, '_diag_encode_step', 0)
        _do_diag = (_is_seqkd and self.accelerator.is_main_process and _diag_step < 3)
        if _is_seqkd:
            self._diag_encode_step = _diag_step + 1

        with self._template_context(template, mode=mode):
            for idx, data in enumerate(inputs):
                if 'response_token_ids' in data and data['response_token_ids']:
                    data['messages'] = replace_assistant_response_with_ids(data['messages'], data['response_token_ids'])

                if encode_prompt_only:
                    messages = data.get('messages', [])
                    if messages and messages[-1].get('role') == 'assistant':
                        messages[-1]['content'] = None

                if _do_diag and idx == 0:
                    msgs = data.get('messages', [])
                    logger.info(f'[DIAG-ENCODE] num_messages={len(msgs)} | keys={list(data.keys())[:10]}')
                    for mi, m in enumerate(msgs[:8]):
                        role = m.get('role', '?')
                        content = m.get('content', '')
                        c_preview = str(content)[:120] if content else '<None>'
                        logger.info(f'[DIAG-ENCODE]   msg[{mi}] role={role} | content({type(content).__name__})={c_preview}')

                encoded = template.encode(data, return_length=True)

                if _do_diag and idx == 0:
                    enc_labels = encoded.get('labels')
                    enc_ids = encoded.get('input_ids')
                    if enc_labels is not None:
                        n_valid = sum(1 for l in enc_labels if l != -100)
                        logger.info(f'[DIAG-ENCODE] encoded: input_ids_len={len(enc_ids)}, '
                                    f'labels_len={len(enc_labels)}, labels_valid={n_valid}')
                    else:
                        logger.info(f'[DIAG-ENCODE] encoded: labels=None!')

                batch_encoded_inputs.append(encoded)

            batch_encoded = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

        return batch_encoded


    @patch_profiling_decorator
    def training_step(self,
                      model: nn.Module,
                      inputs: DataType,
                      num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        args = self.args

        with patch_profiling_context(self, 'get_completions'):
            if self._get_random_num() <= self.lmbda:
                data_source = DataSource.STUDENT

                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)

                if self.toolace_mode and self.toolace_onpolicy_mode == 'full':
                    with unwrap_model_for_generation(
                            model, self.accelerator,
                            gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                        unwrapped_model.eval()

                        batch_results = []
                        for data_item in inputs:
                            result = self.toolace_manager.generate_multi_turn_sequence(
                                data_item=data_item,
                                model=unwrapped_model,
                                tokenizer=self.processing_class,
                                generation_config=self.generation_config,
                                template=self.template,
                                device=self.model.device,
                                max_completion_length=self.max_completion_length,
                            )
                            batch_results.append(result)

                        unwrapped_model.train()

                    encoded_inputs = self._collate_toolace_batch(batch_results)

                elif self.toolace_mode and self.toolace_onpolicy_mode == 'teacher_forcing':

                    if args.use_vllm:
                        processed_inputs = self._preprocess_inputs(inputs)
                        generated_inputs = self._fast_infer(processed_inputs)

                        if self.log_completions:
                            messages = [inp['messages'][:-1] for inp in generated_inputs]
                            completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
                            valid_messages = gather_object(messages)
                            valid_completions = gather_object(completions)
                            self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
                            self._logs['completion'].extend(valid_completions)

                        with self._template_context(self.template):
                            encoded_inputs = self._prepare_batch_inputs(generated_inputs, encode_prompt_only=False)
                    else:
                        encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)

                        with unwrap_model_for_generation(
                                model, self.accelerator,
                                gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                            unwrapped_model.eval()
                            new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                                unwrapped_model, encoded_inputs, self.generation_config,
                                self.processing_class.pad_token_id)
                            unwrapped_model.train()

                        encoded_inputs['input_ids'] = new_input_ids
                        encoded_inputs['attention_mask'] = new_attention_mask
                        encoded_inputs['labels'] = new_labels

                    if self.toolace_mask_tool_response:
                        from data.toolace_env import build_toolace_off_policy_labels
                        encoded_inputs['labels'] = build_toolace_off_policy_labels(
                            input_ids=encoded_inputs['input_ids'],
                            labels=encoded_inputs['labels'],
                            tokenizer=self.processing_class,
                        )

                elif args.use_vllm:
                    processed_inputs = self._preprocess_inputs(inputs)
                    generated_inputs = self._fast_infer(processed_inputs)

                    if self.log_completions:
                        messages = [inp['messages'][:-1] for inp in generated_inputs]
                        completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
                        valid_messages = gather_object(messages)
                        valid_completions = gather_object(completions)
                        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
                        self._logs['completion'].extend(valid_completions)

                    with self._template_context(self.template):
                        encoded_inputs = self._prepare_batch_inputs(generated_inputs, encode_prompt_only=False)
                else:
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)

                    with unwrap_model_for_generation(
                            model, self.accelerator,
                            gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                        unwrapped_model.eval()
                        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                            unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                        unwrapped_model.train()

                    encoded_inputs['input_ids'] = new_input_ids
                    encoded_inputs['attention_mask'] = new_attention_mask
                    encoded_inputs['labels'] = new_labels

            elif self.seq_kd:
                data_source = DataSource.TEACHER

                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)

                encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)

                load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
                with load_context, unwrap_model_for_generation(
                        self.teacher_model,
                        self.accelerator,
                        gather_deepspeed3_params=self.teacher_ds3_gather_for_generation) as unwrapped_model:
                    unwrapped_model.eval()
                    new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                        unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)

                encoded_inputs['input_ids'] = new_input_ids
                encoded_inputs['attention_mask'] = new_attention_mask
                encoded_inputs['labels'] = new_labels

            else:
                data_source = DataSource.DATASET
                total_length = self.template.max_length + self.max_completion_length
                with self._template_context(self.template, max_length=total_length):
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)


            encoded_inputs['_data_source'] = data_source
            if self.accelerator.is_main_process and hasattr(self, 'state') \
                    and self.state.global_step % 10 == 0:
                self._debug_log_student_output(encoded_inputs, data_source)
            if self.gkd_algorithm in ('mad_opd', 'mt_opd'):
                self._run_teacher_interaction(encoded_inputs)

                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                    logger.info(
                        f'[GKD] dist.barrier() after teacher interaction '
                        f'(algorithm={self.gkd_algorithm})'
                    )

            torch.cuda.empty_cache()

        with self.template.forward_context(self.model, encoded_inputs):
            loss = HFSFTTrainer.training_step(self, model, encoded_inputs, num_items_in_batch)
        return loss

    def _extract_prompt_and_response(self, encoded_inputs: Dict[str, torch.Tensor]) -> Tuple[list, list]:
        input_ids = encoded_inputs['input_ids']
        labels = encoded_inputs['labels']
        batch_size = input_ids.shape[0]

        prompts = []
        response_token_ids_list = []
        response_labels_mask_list = []
        prompt_lengths = []
        responses = []

        for i in range(batch_size):
            sample_ids = input_ids[i]
            sample_labels = labels[i]

            prompt_mask = sample_labels == -100
            prompt_ids = sample_ids[prompt_mask]
            prompt_text = self.processing_class.decode(prompt_ids, skip_special_tokens=True).strip()

            response_mask = sample_labels != -100
            response_ids = sample_ids[response_mask]
            response_text = self.processing_class.decode(response_ids, skip_special_tokens=True).strip()

            prompts.append(prompt_text)
            responses.append(response_text)
            response_token_ids_list.append(response_ids.cpu())
            response_labels_mask_list.append(None)

            if 'attention_mask' in encoded_inputs:
                attn_mask = encoded_inputs['attention_mask'][i]
                real_prompt_len = ((attn_mask == 1) & (sample_labels == -100)).sum().item()
            else:
                real_prompt_len = (sample_labels == -100).sum().item()
            prompt_lengths.append(real_prompt_len)

        return prompts, responses, response_token_ids_list, prompt_lengths, response_labels_mask_list


    def _run_teacher_interaction(self, encoded_inputs: Dict[str, torch.Tensor]):
        prompts, responses, response_token_ids_list, prompt_lengths, response_labels_mask_list = self._extract_prompt_and_response(encoded_inputs)
        teacher_ids = encoded_inputs.pop('_teacher_ids', None)
        self._current_prompt_lengths = prompt_lengths
        batch_size = len(prompts)

        if not prompts or not any(prompts):
            return

        try:
            if self.use_vllm_teachers and self.vllm_teacher_manager is not None:
                self._run_vllm_teacher_interaction(
                    prompts, responses,
                    response_token_ids_list=response_token_ids_list,
                    response_labels_mask_list=response_labels_mask_list,
                    teacher_ids=teacher_ids,
                )


        except Exception as exc:
            logger.warning(
                f'[GKD] Teacher interaction failed for {self.gkd_algorithm}, '
                f'falling back to multi_teacher loss. Error: {exc}'
            )

    @torch.no_grad()
    def _run_vllm_teacher_interaction(self, prompt_texts: list, response_texts: list,
                                       response_token_ids_list: list = None,
                                       response_labels_mask_list: list = None, teacher_ids: list = None):
        manager = self.vllm_teacher_manager
        batch_size = len(prompt_texts)

        all_prompt_lists = manager.allgather_texts(prompt_texts)
        all_response_lists = manager.allgather_texts(response_texts)
        all_prompts = [p for rank_prompts in all_prompt_lists for p in rank_prompts]
        all_responses = [r for rank_responses in all_response_lists for r in rank_responses]

        #=== AllGather response token IDs ===
        all_response_token_ids = None
        if response_token_ids_list is not None:
            local_ids_as_lists = [ids.tolist() for ids in response_token_ids_list]
            all_ids_lists = manager.allgather_texts(local_ids_as_lists)  # reuse allgather
            all_response_token_ids = [ids for rank_ids in all_ids_lists for ids in rank_ids]


        all_responses, repetition_mask = self._filter_repetitive_responses(all_responses)


        my_local_rank = manager.local_rank
        start_idx = my_local_rank * batch_size
        end_idx = start_idx + batch_size

        if self.gkd_algorithm == 'mad_opd':
            # MAD-OPD (Paper §4): K teachers debate for R rounds WITHOUT seeing
            # the student reply ("blind" debate), so the debate transcript H is
            # genuine privileged information and ``dynamic_weights`` applies the
            # confidence-based reweighting of Paper Eq. 6-7.
            full_results = manager.batch_blind_mad_debate_and_force_decode(
                all_prompts=all_prompts,
                all_responses=all_responses,
                all_response_token_ids=all_response_token_ids,
                rounds=self.mad_debate_rounds,
                t1_max_tokens=self.mad_t1_max_tokens,
                t2_max_tokens=self.mad_t2_max_tokens,
                temperature=self.mad_debate_temperature,
                top_p=self.mad_debate_top_p,
                top_k=self.mad_debate_top_k,
                device=str(self.model.device),
                toolace_mode=getattr(self, 'toolace_mode', False),
                code_mode=getattr(self, 'code_mode', False),
                dynamic_weights=self.mad_dynamic_weights,
            )
        elif self.gkd_algorithm == 'mt_opd':
            # MT-OPD baseline (Paper §5.1): multi-teacher distillation WITHOUT
            # debate.  Both teachers force-decode the student's on-policy output
            # in parallel; losses are combined with fixed ``teacher_weights``.
            all_t1_logits, all_t2_logits = manager.force_decode_parallel(
                context_texts_1=list(all_prompts),
                student_output_texts_1=all_responses,
                context_texts_2=list(all_prompts),
                student_output_texts_2=all_responses,
                device=str(self.model.device),
                student_output_token_ids=all_response_token_ids,
            )
            alpha_f, beta_f = (self.teacher_weights + [0.5, 0.5])[:2]
            full_results = {
                'teacher_a_logits': all_t1_logits,
                'teacher_b_logits': all_t2_logits,
                'weights': [(alpha_f, beta_f)] * len(all_prompts),
                'debate_histories': [''] * len(all_prompts),
            }
        else:
            full_results = None

        if self.gkd_algorithm in ('mad_opd', 'mt_opd') and full_results is not None:
            device = str(self.model.device)
            my_batch_results = []

            max_logits_bytes = getattr(self.args, 'max_logits_bytes', 6 * 1024 * 1024 * 1024)

            for idx in range(start_idx, end_idx):
                teacher_a_logits = full_results['teacher_a_logits'][idx]
                teacher_b_logits = full_results['teacher_b_logits'][idx]
                logits_size_a = teacher_a_logits.nelement() * teacher_a_logits.element_size() if teacher_a_logits is not None else 0
                logits_size_b = teacher_b_logits.nelement() * teacher_b_logits.element_size() if teacher_b_logits is not None else 0
                if logits_size_a > max_logits_bytes or logits_size_b > max_logits_bytes:
                    logger.warning(
                        f'[GKD] Sample {idx}: logits too large '
                        f'(A={logits_size_a / 1e9:.2f}GB, B={logits_size_b / 1e9:.2f}GB), '
                        f'skipping to prevent OOM (limit={max_logits_bytes / 1e9:.1f}GB)'
                    )
                    teacher_a_logits = None
                    teacher_b_logits = None

                if teacher_a_logits is not None:
                    teacher_a_logits = teacher_a_logits.to(device)
                if teacher_b_logits is not None:
                    teacher_b_logits = teacher_b_logits.to(device)
                alpha, beta = full_results['weights'][idx]
                my_batch_results.append({
                    'teacher_a_logits': teacher_a_logits,
                    'teacher_b_logits': teacher_b_logits,
                    'alpha': alpha,
                    'beta': beta,
                    'debate_history': full_results.get('debate_histories', [''] * len(all_prompts))[idx],
                })
            for i in range(len(full_results['teacher_a_logits'])):
                full_results['teacher_a_logits'][i] = None
                full_results['teacher_b_logits'][i] = None

            self._current_batch_vllm_results = my_batch_results
            self._current_vllm_logits_ready = True

            # === Teacher Review Generation (MAD only) ===
            if self.gkd_algorithm in ('mad_opd',):
                self._maybe_generate_teacher_review(
                    manager=manager,
                    all_prompts=all_prompts,
                    all_responses=all_responses,
                    full_results=full_results,
                    start_idx=start_idx,
                )

    def _filter_repetitive_responses(
        self,
        responses: List[str],
        ngram_size: int = 3,
        max_repetition_ratio: float = 0.6,
    ) -> Tuple[List[str], List[bool]]:
        """
        Detect and filter repetitive (looping) student responses.

        A response is considered repetitive if its n-gram repetition ratio
        exceeds max_repetition_ratio. Repetitive responses are replaced with
        empty strings, causing Sidecar to return None logits (automatically
        skipped in loss computation).

        Args:
            responses: List of student response strings
            ngram_size: N-gram size for repetition detection (default: 3)
            max_repetition_ratio: Maximum allowed ratio of repeated n-grams.
                0.0 = all unique, 1.0 = all repeated. Default 0.5.

        Returns:
            Tuple of (filtered_responses, is_valid_mask)
        """
        filtered = []
        is_valid = []

        for i, response in enumerate(responses):
            # N-gram repetition ratio check
            words = response.lower().split()
            if len(words) < ngram_size:
                filtered.append(response)
                is_valid.append(True)
                continue

            ngrams = set()
            total = 0
            for ng in zip(*[words[j:] for j in range(ngram_size)]):
                ngrams.add(ng)
                total += 1

            repetition_ratio = 1.0 - len(ngrams) / total if total > 0 else 0.0
            if repetition_ratio > max_repetition_ratio:
                logger.warning(
                    f'[GKD] Sample {i}: high repetition ratio ({repetition_ratio:.2%}, '
                    f'{len(words)} words), replacing with empty string to skip distillation'
                )
                filtered.append('')
                is_valid.append(False)
                continue

            filtered.append(response)
            is_valid.append(True)

        num_filtered = sum(1 for v in is_valid if not v)
        if num_filtered > 0:
            logger.info(
                f'[GKD] Filtered {num_filtered}/{len(responses)} repetitive responses '
                f'(ngram_size={ngram_size}, max_ratio={max_repetition_ratio})'
            )

        return filtered, is_valid

    def _collate_toolace_batch(self, batch_results):
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id

        max_length = max(result['input_ids'].shape[1] for result in batch_results)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for result in batch_results:
            seq_len = result['input_ids'].shape[1]
            pad_length = max_length - seq_len

            if pad_length > 0:
                input_ids = F.pad(result['input_ids'], (pad_length, 0), value=pad_token_id)
                attention_mask = F.pad(result['attention_mask'], (pad_length, 0), value=0)
                labels = F.pad(result['labels'], (pad_length, 0), value=-100)
            else:
                input_ids = result['input_ids']
                attention_mask = result['attention_mask']
                labels = result['labels']

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        collated = {
            'input_ids': torch.cat(batch_input_ids, dim=0),
            'attention_mask': torch.cat(batch_attention_mask, dim=0),
            'labels': torch.cat(batch_labels, dim=0),
        }

        collated['position_ids'] = collated['attention_mask'].cumsum(dim=1) - 1
        collated['position_ids'][collated['position_ids'] < 0] = 0

        return collated

    def prediction_step(self, model, inputs, *args, **kwargs):
        encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)
        with self.template.forward_context(self.model, encoded_inputs):
            return super().prediction_step(model, encoded_inputs, *args, **kwargs)

    @contextmanager
    def offload_context(self):
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()

        try:
            yield
        finally:
            if self.args.offload_model:
                self.load_model(self.accelerator.unwrap_model(self.model))
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                self.load_optimizer()

    def _get_random_num(self) -> float:
        seed = int(getattr(self.args, 'seed', 0))
        seed += int(self.state.global_step)
        rng = random.Random(seed)
        return rng.random()

    @contextmanager
    def load_teacher_model_context(self):
        if not self.args.offload_teacher_model:
            yield
            return

        self.load_model(self.accelerator.unwrap_model(self.teacher_model))
        yield
        self.offload_model(self.accelerator.unwrap_model(self.teacher_model))

    @contextmanager
    def load_both_teacher_models_context(self):
        if self.use_vllm_teachers:
            yield
            return

        if not self.args.offload_teacher_model:
            yield
            return

        self.load_model(self.accelerator.unwrap_model(self.teacher_model))
        if self.teacher_model_2 is not None:
            self.load_model(self.accelerator.unwrap_model(self.teacher_model_2))
        try:
            yield
        finally:
            self.offload_model(self.accelerator.unwrap_model(self.teacher_model))
            if self.teacher_model_2 is not None:
                self.offload_model(self.accelerator.unwrap_model(self.teacher_model_2))

    def _get_teacher_for_generation(self, teacher_model):
        if self.use_vllm_teachers:
            logger.warning('[GKD] _get_teacher_for_generation called in vLLM teacher mode, this is unexpected')
            return teacher_model
        elif self.is_teacher_ds3:
            return teacher_model
        else:
            return self.accelerator.unwrap_model(teacher_model)

    def _prepare_liger_loss(self):
        args = self.args
        self.use_liger_gkd_loss = False

        if getattr(args, 'use_liger_kernel', False):
            if not _liger_kernel_available:
                raise ImportError(
                    'Liger kernel is not installed. Please install liger-kernel by running: pip install liger-kernel')
            assert self.args.sft_alpha == 0, 'SFT loss is not supported with liger loss'

            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=self.beta,
                ignore_index=-100,
                temperature=self.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        chunk_size=512,
    ):
    
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if labels is not None:
            mask = labels != -100
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
            num_valid = mask.sum()
        else:
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            num_valid = student_logits.size(0)

        if num_valid == 0:
            return student_logits.new_zeros(())

        num_valid_int = num_valid if isinstance(num_valid, int) else num_valid.item()
        total_loss = student_logits.new_zeros(())

        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        for start_idx in range(0, num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid_int)
            s_chunk = student_logits[start_idx:end_idx]
            t_chunk = teacher_logits[start_idx:end_idx]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            del s_chunk, t_chunk

            if beta == 0:
                jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
            elif beta == 1:
                jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
            else:
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )

                kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
                kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
                del mixture_log_probs

                # JSD = β * KL(M||Q) + (1-β) * KL(M||P)
                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
                del kl_teacher, kl_student

            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs

        return total_loss / num_valid

    def _maybe_generate_teacher_review(
        self,
        manager,
        all_prompts: List[str],
        all_responses: List[str],
        full_results: dict,
        start_idx: int,
    ):
        """
        Conditionally generate teacher review texts for human review.

        Every `teacher_review_interval` training steps, picks one sample from the batch,
        sends the original prompt + debate history to both teacher vLLM servers for
        free-form text generation, and logs the results to teacher_review_generations.jsonl.

        IMPORTANT: This method must be called by ALL ranks (not just main_rank), because
        manager.generate() internally uses broadcast to distribute results from main_rank
        to all other ranks. If only main_rank calls generate(), other ranks would deadlock
        waiting for the broadcast. The JSONL writing is guarded by is_main_rank check.

        Design:
        - All ranks call manager.generate() (broadcast-safe)
        - Only main_rank writes to JSONL and logs
        - Wrapped in try/except so failures never interrupt training
        - Uses low temperature (0.3) for deterministic, reviewable outputs

        Args:
            manager: VLLMTeacherManager instance
            all_prompts: All prompts across ranks (after AllGather)
            all_responses: All student responses across ranks (after AllGather)
            full_results: Results dict from batch_mad/blind_mad_debate_and_force_decode,
                          containing 'debate_histories' key
            start_idx: Start index for current rank's batch slice
        """
        if self.teacher_review_interval <= 0 or self.teacher_review_writer is None:
            return

        current_step = getattr(self.state, 'global_step', 0) if hasattr(self, 'state') else 0
        if current_step <= 0 or current_step % self.teacher_review_interval != 0:
            return

        try:
            sample_idx = start_idx
            prompt = all_prompts[sample_idx] if sample_idx < len(all_prompts) else ''
            student_response = all_responses[sample_idx] if sample_idx < len(all_responses) else ''

            debate_history = ''
            if full_results and 'debate_histories' in full_results:
                debate_history = full_results['debate_histories'][sample_idx] or ''

            review_prompt = (
                f'{prompt}\n\n'
                f'--- Teacher Debate History ---\n{debate_history}\n\n'
                f'Based on the above debate, please provide your final answer to the original question.'
            )

            # All ranks must call generate() because it contains internal broadcast.
            # main_rank sends HTTP request, other ranks wait for broadcast result.
            teacher_1_review = manager.generate(
                engine_name='teacher_1',
                prompts=[review_prompt],
                max_tokens=getattr(self, 'mad_t1_max_tokens', 4096),
                temperature=0.3,
                top_p=0.9,
            )

            teacher_2_review = manager.generate(
                engine_name='teacher_2',
                prompts=[review_prompt],
                max_tokens=getattr(self, 'mad_t2_max_tokens', 4096),
                temperature=0.3,
                top_p=0.9,
            )

            # Only main_rank writes to JSONL to avoid duplicate entries
            if manager.is_main_rank:
                teacher_1_text = teacher_1_review[0] if teacher_1_review else ''
                teacher_2_text = teacher_2_review[0] if teacher_2_review else ''

                review_record = {
                    'step': current_step,
                    'prompt': prompt,
                    'student_response': student_response[:3000],
                    'debate_history': debate_history[:8000],
                    'teacher_1_review': teacher_1_text,
                    'teacher_2_review': teacher_2_text,
                }
                self.teacher_review_writer.append(review_record)

                logger.info(
                    f'[Teacher Review] Step {current_step} | '
                    f'T1 ({len(teacher_1_text)} chars): {teacher_1_text[:300]}... | '
                    f'T2 ({len(teacher_2_text)} chars): {teacher_2_text[:300]}...'
                )
        except Exception as exc:
            logger.warning(f'[Teacher Review] Failed at step {current_step}: {exc}')

    def _prepare_logging(self):
        args = self.args
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = getattr(args, 'wandb_log_unique_prompts', False)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))

        # Teacher review generation: periodically let teachers generate text for human review
        self.teacher_review_interval = getattr(args, 'teacher_review_interval', 0)
        if self.teacher_review_interval > 0:
            self.teacher_review_writer = JsonlWriter(
                os.path.join(self.args.output_dir, 'teacher_review_generations.jsonl')
            )
            logger.info(
                f'[Teacher Review] Enabled: generating teacher review texts every '
                f'{self.teacher_review_interval} steps, saving to teacher_review_generations.jsonl'
            )
        else:
            self.teacher_review_writer = None

        self._logs = {
            'prompt': deque(),
            'completion': deque(),
        }

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log method to include completion table logging (aligned with GRPO)."""
        # Aggregate custom MAD-OPD metrics into the training logs.
        mode = 'train' if self.model.training else 'eval'
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == 'eval':
            metrics = {f'eval_{key}': val for key, val in metrics.items()}
        logs.update(metrics)

        # Call parent log method
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Clear metrics after logging
        self._metrics[mode].clear()

        # Log completions table if we have data (only for on-policy generations)
        if self.accelerator.is_main_process and self.log_completions and len(self._logs['prompt']) > 0:
            seen_nums = len(self._logs['prompt'])
            table = {
                'step': [str(self.state.global_step)] * seen_nums,
                'prompt': list(self._logs['prompt'])[:seen_nums],
                'completion': list(self._logs['completion'])[:seen_nums],
            }

            # Write to jsonl
            self.jsonl_writer.append(table)

            # Log to wandb if enabled
            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            if report_to_wandb:
                import pandas as pd
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

            # Log to swanlab if enabled
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab.get_run(
            ) is not None
            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = [table[header][i] for header in headers]
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})

            # Clear logs after all logging is done
            self._logs['prompt'].clear()
            self._logs['completion'].clear()


# Backwards-compatibility alias for the ms-swift registry.
GKDTrainer = MADOPDTrainer
