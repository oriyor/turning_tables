import json
import logging
import copy
import os
import pandas as pd
from dataclasses import dataclass
from datasets import Dataset, tqdm
from torch.cuda import amp
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import default_data_collator
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict, Optional
import torch
from transformers.trainer_pt_utils import LengthGroupedSampler, get_length_grouped_indices
from ContinuousPreTraining.Common.config import Config
from ContinuousPreTraining.Data.datasets_wrapper import DatasetsWrapper
from ContinuousPreTraining.Training.trainers.basic_trainer import BasicTrainer
from ContinuousPreTraining.basic_factory import BasicFactory
import numpy as np

#####  --- transformer trainer train() --- imports START #####

import collections
import inspect
import math
import os
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    init_deepspeed,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.trainer_utils import TrainOutput, speed_metrics

from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_distributed_available,
    is_torch_tpu_available,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments

# from transformers.utils import logging
from ContinuousPreTraining.predictors.span_predictor import SpanPrediction

_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# if is_apex_available():
#     from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_distributed_available():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if TYPE_CHECKING:
    import optuna

#####  --- transformer trainer train() --- imports END #####

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


# todo move to a different place
### to move ###
def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
    # tood pad value by tokenizer
    return seq + (max_batch_len - len(seq)) * [pad_value]


@dataclass
class DynamicInputIdsPaddingDataCollator:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """
    pad_token_id: int
    train_mode: bool = True

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch_inputs = list()
        batch_attention_masks = list()
        labels = list()
        max_size = max([len(ex['input_ids']) for ex in examples])

        # add pads and attention mask to mask size in batch
        for item in examples:
            batch_inputs += [pad_seq(item['input_ids'], max_size, 0)]
            labels.append(item['labels'])
            batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]

        batch = {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
                 "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
                 "labels": torch.stack(labels)
                 }

        # if we are not in train mode, we want to have all the dataset features
        # this is based on code from default data collator
        if not self.train_mode:
            first = examples[0]
            for k, v in first.items():
                if k not in ("labels", "input_ids", "attention_mask"):
                    batch[k] = [f[k] for f in examples]

        return batch

@dataclass
class IdCollator:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """
    pad_token_id: int
    train_mode: bool = True

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        return examples

class LengthGroupedSamplerWithLargeMegabatches(LengthGroupedSampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    This is similar to LengthGroupedSampler, but we want the number of mega batches to be small, and not have a minimum of 50
    examples in a megabatch
    """

    def __iter__(self):
        # set megabatch multi to always be 1/4 of the batches
        mega_batch_mult = len(self.lengths) // (self.batch_size * 4)

        # for tiny datasets, mega_batch_mult has to be at least 1
        mega_batch_mult = max(1, mega_batch_mult)

        indices = get_length_grouped_indices(self.lengths,
                                             self.batch_size,
                                             mega_batch_mult=mega_batch_mult,
                                             )

        # next, shuffle the batches such that we won't starve batches with short length
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        import random
        # we want to keep the longest batch first and the last one with the leftover to avoid bugs and OOM errors
        if len(batches) > 3:
            batches_copy = batches[1:-1]

            # use torch for shuffling, random.randomseed may not have been init yet
            shuffled_batches_copy_indices = torch.randperm(len(batches) - 2)
            batches_copy = [batches_copy[i] for i in shuffled_batches_copy_indices]
            batches[1:-1] = batches_copy  # overwrite the original

        shuffled_indices = sum(batches, [])

        return iter(shuffled_indices)


## end of to move ###
class SubtaskDataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    # TODO Think if there's a better way to init the dataloader
    # add an option to reset a dataset
    # call it on stop iteration
    def __init__(self,
                 task_name,
                 dataset,
                 dataloader_args,
                 trainer_args,
                 tokenizer,
                 is_train_task,
                 heterogeneous_sampling=False
                 ):

        # we save all the info in fields because we want to re-init the dataloader on stop iteration
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.dataloader_args = dataloader_args
        self.drop_last = trainer_args.dataloader_drop_last
        self.dataloader_num_workers = trainer_args.dataloader_num_workers
        self.is_train_task = is_train_task

        # set batch size
        # check if this is a train/eval task and whether we should use the default train/eval batch size
        self.default_batch_size = trainer_args.per_device_train_batch_size if self.is_train_task else trainer_args.per_device_eval_batch_size
        self.batch_size = self.default_batch_size if 'batch_size' not in self.dataloader_args else self.dataloader_args[
            'batch_size']

        # batch size must be one for heterogeneous samplign
        if heterogeneous_sampling:
            self.batch_size = 1

        logger.info(f'Batch size for task: {self.task_name}, train mode: {self.is_train_task}, '
                    f'is: {self.batch_size}')

        # we'll init the dataloader using an init_dataloader method,
        # which we'll also be used from the outside method
        self.data_loader = None
        # get sampler and collator
        # we support two options, either default collator, or dynamic padding with grouped length sampler
        self.sampler, self.collator = self.get_sampler_and_collator(dataloader_args.get('single_task_sampler',
                                                                                        'Random'))

        self.init_dataloader()

    def get_sampler_and_collator(self, sampler_name):
        """
        get the sampler and collator that will be used for a single torch dataloader
        """
        if sampler_name == 'LengthGroupedSampler':

            # we will use LengthGroupedSampler with dynamic padding
            logger.info(f'Using length group sampling for task {self.task_name}')

            # checking if we already have lengths in the dataset so that we won't iterate the dataset
            if hasattr(self.dataset, 'lengths'):
                logger.info('Using lengths from dataset')
                lengths = self.dataset.lengths

            # else, calculate lenghts
            else:
                logger.info('Calculating input lengths for fast training')
                lengths = [len(self.dataset[i]['input_ids'])
                           for i in tqdm(range(len(self.dataset)))]
            # check if we need to calculate the lengths

            sampler = LengthGroupedSamplerWithLargeMegabatches(self.dataset,
                                                               self.batch_size,
                                                               lengths=lengths)
            collator = IdCollator(self.tokenizer)
            return sampler, collator

        else:
            # todo for eval the sampler should be sequential
            sampler = RandomSampler(self.dataset)
            collator = default_data_collator
            return sampler, collator

    def init_dataloader(self):
        """
        init the dataloader
        """
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      collate_fn=self.collator,
                                      sampler=self.sampler,
                                      drop_last=self.drop_last,
                                      num_workers=self.dataloader_num_workers,
                                      )

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            # todo to change optimization for task, we need to return the task name
            # catch it in the trainer and update the optimization
            # for example, set the LR to be self.state['task_name']['lr'] before doing a step

            # only pass task name when overriding HF train
            # if Config().get('trainer.override_huggingface_train_method'):
            #     batch["task_name"] = self.task_name

            yield batch


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    # TODO Think if there's a better way to init the dataloader
    # add an option to reset a dataset
    # call it on stop iteration
    def __init__(self,
                 task_name,
                 dataset,
                 dataloader_args,
                 trainer_args,
                 tokenizer,
                 is_train_task
                 ):

        # we save all the info in fields because we want to re-init the dataloader on stop iteration
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.dataloader_args = dataloader_args
        self.drop_last = trainer_args.dataloader_drop_last
        self.dataloader_num_workers = trainer_args.dataloader_num_workers
        self.is_train_task = is_train_task

        # set batch size
        # check if this is a train/eval task and whether we should use the default train/eval batch size
        self.default_batch_size = trainer_args.per_device_train_batch_size if self.is_train_task else trainer_args.per_device_eval_batch_size
        self.batch_size = self.default_batch_size if 'batch_size' not in self.dataloader_args else self.dataloader_args[
            'batch_size']
        logger.info(f'Batch size for task: {self.task_name}, train mode: {self.is_train_task}, '
                    f'is: {self.batch_size}')

        # we'll init the dataloader using an init_dataloader method,
        # which we'll also be used from the outside method
        self.data_loader = None
        # get sampler and collator
        # we support two options, either default collator, or dynamic padding with grouped length sampler
        self.sampler, self.collator = self.get_sampler_and_collator(dataloader_args.get('single_task_sampler',
                                                                                        'Random'))

        self.init_dataloader()

    def get_sampler_and_collator(self, sampler_name):
        """
        get the sampler and collator that will be used for a single torch dataloader
        """
        if sampler_name == 'LengthGroupedSampler':

            # we will use LengthGroupedSampler with dynamic padding
            logger.info(f'Using length group sampling for task {self.task_name}')

            # checking if we already have lengths in the dataset so that we won't iterate the dataset
            if hasattr(self.dataset, 'lengths'):
                logger.info('Using lengths from dataset')
                lengths = self.dataset.lengths

            # else, calculate lenghts
            else:
                logger.info('Calculating input lengths for fast training')
                lengths = [len(self.dataset[i]['input_ids'])
                           for i in tqdm(range(len(self.dataset)))]
            # check if we need to calculate the lengths

            sampler = LengthGroupedSamplerWithLargeMegabatches(self.dataset,
                                                               self.batch_size,
                                                               lengths=lengths)
            collator = DynamicInputIdsPaddingDataCollator(self.tokenizer)
            return sampler, collator

        else:
            # todo for eval the sampler should be sequential
            sampler = RandomSampler(self.dataset)
            collator = default_data_collator
            return sampler, collator

    def init_dataloader(self):
        """
        init the dataloader
        """
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      collate_fn=self.collator,
                                      sampler=self.sampler,
                                      drop_last=self.drop_last,
                                      num_workers=self.dataloader_num_workers,
                                      )

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            # todo to change optimization for task, we need to return the task name
            # catch it in the trainer and update the optimization
            # for example, set the LR to be self.state['task_name']['lr'] before doing a step

            # only pass task name when overriding HF train
            if Config().get('trainer.override_huggingface_train_method'):
                batch["task_name"] = self.task_name

            yield batch


def reformat_multi_dataset_task_for_recursive_call(multi_sub_task_datasets_wrapper):
    """
    receives a datasets_wrapper with multiple datasets and returns multiple single_datset wrappers
    for the recursive call to MultitaskDataloader
    """
    # todo prettify this, should we have a list of dataset wrappers of size 1? (but then where will we put the sampler?)
    # initial a dict between each subtask and a dataset wrapper with just 1 datsaet
    sub_task_to_dataset_wrapper_dict = {}
    for i, dataset in enumerate(multi_sub_task_datasets_wrapper.datasets):
        sub_task_name = multi_sub_task_datasets_wrapper.datasets_names[i]
        sub_task_dataset_wrapper = DatasetsWrapper(datasets=[dataset],
                                                   dataloader_args=multi_sub_task_datasets_wrapper.dataloader_args,
                                                   datasets_names=[sub_task_name],
                                                   sampler=multi_sub_task_datasets_wrapper.sampler,
                                                   predictor=multi_sub_task_datasets_wrapper.predictor,
                                                   eval_method=multi_sub_task_datasets_wrapper.eval_method
                                                   )
        sub_task_to_dataset_wrapper_dict[sub_task_name] = sub_task_dataset_wrapper

    # return a dict that can be as a recursive call to MultitaskDataloader
    return sub_task_to_dataset_wrapper_dict


# todo think if we can do this by inheretence
class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self,
                 train_dataset,
                 sampler_config,
                 trainer_state,
                 trainer_args,
                 tokenizer):
        """
        init the dataloaders for each task
        get the sampling indices between the tasks
        """
        self.train_dataset = train_dataset

        # get the sampler from the config, and init it
        self.sampler_config = copy.deepcopy(sampler_config)
        sampler_name = self.sampler_config['type']
        del self.sampler_config['type']
        sampler_cls = BasicFactory().get_object(sampler_name)
        if 'pass_trainer_state' in self.sampler_config:
            del self.sampler_config['pass_trainer_state']
            self.sampler_config['trainer_state'] = trainer_state
            self.sampler = sampler_cls(**self.sampler_config)
        else:
            self.sampler = sampler_cls(**self.sampler_config)

        self.tokenizer = tokenizer
        self.task_name_list = list(self.train_dataset)
        self.num_examples = sum([datasets_wrapper.num_examples
                                 for datasets_wrapper in self.train_dataset.values()])

        # init an iterable for every task
        self.task_iterables = {}

        for task_name, task_datasets_wrapper in self.train_dataset.items():

            logger.info(f'Initializing data iterable for task {task_name}')

            # init a dictionary for very task, we'll populate this with length and iterable fields
            self.task_iterables[task_name] = {}

            # if this is a single dataset
            if task_datasets_wrapper.single_dataset:
                task_dataloader = DataLoaderWithTaskname(task_name,
                                                         task_datasets_wrapper.datasets[0],
                                                         task_datasets_wrapper.dataloader_args,
                                                         trainer_args,
                                                         self.tokenizer,
                                                         is_train_task=task_datasets_wrapper.is_train_task
                                                         )
                self.task_iterables[task_name]['dataloader'] = task_dataloader
                self.task_iterables[task_name]['iterable'] = iter(task_dataloader)
                self.task_iterables[task_name]['num_batches'] = len(task_dataloader)

            # if this are multiple datasets
            else:
                # reformat the multi dataset wrapper to a dict
                reformatted_multi_dataset_task = reformat_multi_dataset_task_for_recursive_call(task_datasets_wrapper)

                # get the sub task sampler
                sub_task_sampler_config = task_datasets_wrapper.sampler

                # recursively call the MultitaskDataloader
                task_dataloader = HeterogeneousMultitaskDataloader(reformatted_multi_dataset_task,
                                                      sub_task_sampler_config,
                                                      trainer_state,
                                                      trainer_args,
                                                      tokenizer)
                self.task_iterables[task_name]['dataloader'] = task_dataloader
                self.task_iterables[task_name]['iterable'] = iter(task_dataloader)
                self.task_iterables[task_name]['num_batches'] = len(task_dataloader)

        # after we inited all the datasets, we can get the sampling indices
        self.task_indices = self.sampler.sample(self.task_iterables)

    def __len__(self):
        return len(self.task_indices)

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        """
        # get the indices for each iterable
        # iterate over the indices and sample from the iterables
        for i, task_choice in enumerate(self.task_indices):
            task_name = self.task_name_list[task_choice]
            task_generator = self.task_iterables[task_name]['iterable']

            #
            try:
                yield next(task_generator)

            except StopIteration:

                # if we receive a StopItertion from a task iterator, re-init it
                # this will cause random sampling from the dataloader
                num_task_batches = self.task_iterables[task_name]['num_batches']
                logger.info(f'Restarting task {task_name} after stop iteration.'
                            f'Task {task_name} includes {num_task_batches} batches.')
                task_dataloader = self.task_iterables[task_name]['dataloader']
                # task_dataloader.init_dataloader()
                self.task_iterables[task_name]['iterable'] = iter(task_dataloader)

                # if we re-inited the task, we still need to sample from it
                # we know that the length of each task is at least one, so need to try/catch here
                task_generator = self.task_iterables[task_name]['iterable']
                yield next(task_generator)

class HeterogeneousMultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self,
                 train_dataset,
                 sampler_config,
                 trainer_state,
                 trainer_args,
                 tokenizer):
        """
        init the dataloaders for each task
        get the sampling indices between the tasks
        """
        self.train_dataset = train_dataset

        # get the sampler from the config, and init it
        self.sampler_config = copy.deepcopy(sampler_config)
        sampler_name = self.sampler_config['type']
        del self.sampler_config['type']
        sampler_cls = BasicFactory().get_object(sampler_name)
        if 'pass_trainer_state' in self.sampler_config:
            del self.sampler_config['pass_trainer_state']
            self.sampler_config['trainer_state'] = trainer_state
            self.sampler = sampler_cls(**self.sampler_config)
        else:
            self.sampler = sampler_cls(**self.sampler_config)

        self.tokenizer = tokenizer
        self.task_name_list = list(self.train_dataset)
        self.num_examples = sum([datasets_wrapper.num_examples
                                 for datasets_wrapper in self.train_dataset.values()])

        # init an iterable for every task
        self.task_iterables = {}

        for task_name, task_datasets_wrapper in self.train_dataset.items():

            logger.info(f'Initializing data iterable for task {task_name}')

            # init a dictionary for very task, we'll populate this with length and iterable fields
            self.task_iterables[task_name] = {}

            # if this is a single dataset
            if task_datasets_wrapper.single_dataset:

                task_dataloader = SubtaskDataLoaderWithTaskname(task_name,
                                                         task_datasets_wrapper.datasets[0],
                                                         task_datasets_wrapper.dataloader_args,
                                                         trainer_args,
                                                         self.tokenizer,
                                                         is_train_task=task_datasets_wrapper.is_train_task,
                                                         heterogeneous_sampling=True
                                                         )
                self.task_iterables[task_name]['dataloader'] = task_dataloader
                self.task_iterables[task_name]['iterable'] = iter(task_dataloader)
                self.task_iterables[task_name]['num_batches'] = len(task_dataloader)

            # if this are multiple datasets
            else:
                # reformat the multi dataset wrapper to a dict
                reformatted_multi_dataset_task = reformat_multi_dataset_task_for_recursive_call(task_datasets_wrapper)

                # get the sub task sampler
                sub_task_sampler_config = task_datasets_wrapper.sampler

                # recursively call the MultitaskDataloader
                task_dataloader = MultitaskDataloader(reformatted_multi_dataset_task,
                                                      sub_task_sampler_config,
                                                      trainer_state,
                                                      trainer_args,
                                                      tokenizer)
                self.task_iterables[task_name]['dataloader'] = task_dataloader
                self.task_iterables[task_name]['iterable'] = iter(task_dataloader)
                self.task_iterables[task_name]['num_batches'] = len(task_dataloader)

        # after we inited all the datasets, we can get the sampling indices
        self.errors_distribution = self.sampler.sample(self.task_iterables)
        self.batch_size = Config().get('training_arguments.per_device_train_batch_size')
        self.tasks_indices = np.arange(len(self.task_iterables))
        self.num_batches = sum([task['num_batches'] for task in self.task_iterables.values()])

    def update_sampler_with_trainer_state(self, trainer_state):
        """
        update the sampler
        """
        self.sampler.update_sampler_trainer_state(trainer_state)
        self.errors_distribution = self.sampler.sample(self.task_iterables)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        """
        # get the indices for each iterable
        # iterate over the indices and sample from the iterables
        # todo get batch size and distribution
        #
        for i in range(self.num_batches):

            task_names = []
            batch = []
            for _ in range(self.batch_size):
                task_choice = np.random.choice(self.tasks_indices, p=self.errors_distribution)
                task_name = self.task_name_list[task_choice]

                try:
                    task_generator = self.task_iterables[task_name]['iterable']
                    task_names.append(task_name)
                    batch.append(next(task_generator))

                except StopIteration:

                    # if we receive a StopItertion from a task iterator, re-init it
                    # this will cause random sampling from the dataloader
                    num_task_batches = self.task_iterables[task_name]['num_batches']
                    logger.info(f'Restarting task {task_name} after stop iteration.'
                                f'Task {task_name} includes {num_task_batches} batches.')
                    task_dataloader = self.task_iterables[task_name]['dataloader']
                    # task_dataloader.init_dataloader()
                    self.task_iterables[task_name]['iterable'] = iter(task_dataloader)

                    # if we re-inited the task, we still need to sample from it
                    # we know that the length of each task is at least one, so need to try/catch here
                    task_generator = self.task_iterables[task_name]['iterable']
                    batch.append(next(task_generator))
            batch = DynamicInputIdsPaddingDataCollator(self.tokenizer).__call__([example[0] for example in batch])
            batch['task_name'] = collections.Counter(task_names)
            yield batch


class UpdatedMtTrainer(BasicTrainer):
    """
    Trainer for multi task training
    """

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` .

        Because we are using a multi task dataset, return the length of the multi task dataloader
        """
        return dataloader.num_examples

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """

        # get the train sampler
        train_datasets_sampler_config = Config().get('datasets_sampler')

        # use none sampler if no sampler is specified
        if train_datasets_sampler_config is None:
            train_datasets_sampler_config = {"type": 'RandomSampler'}

        # set num examples in state for later used
        return MultitaskDataloader(self.train_dataset,
                                   train_datasets_sampler_config,
                                   self.state,
                                   self.args,
                                   self.tokenizer)

    def update_max_eval_metrics(self, results_dict):
        """
        add a log for the max for every eval metric
        """
        # make sure we have an attribute in the state to save the max metrics
        if not hasattr(self.state, "max_eval_metrics"):
            self.state.max_eval_metrics = {}

        # update trainer state
        for eval_metric_key, eval_metric_value in results_dict.items():
            # populate the field in the trainer state
            if eval_metric_key not in self.state.max_eval_metrics:
                self.state.max_eval_metrics[eval_metric_key] = eval_metric_value
            else:
                self.state.max_eval_metrics[eval_metric_key] = max(self.state.max_eval_metrics[eval_metric_key],
                                                                   eval_metric_value)

        # update result dict from trainer state
        max_metric_suffix = '_max'
        eval_metric_max_keys_map ={eval_metric_key: eval_metric_key + max_metric_suffix
                                   for eval_metric_key in results_dict.keys()}

        for state_dict_key, result_dict_key in eval_metric_max_keys_map.items():
            results_dict[result_dict_key] = self.state.max_eval_metrics[state_dict_key]

        # return the updated dict
        return results_dict

    def evaluate_datasets(self, datasets, train_mode):
        """
        evaluate a dictionary of datasets
        if train mode sample random 1000 examples
        if eval, iterate over all examples
        """
        # this should be in eval mode
        self.model.eval()
        with torch.no_grad():

            for task_name, datasets_wrapper in datasets.items():

                logger.info(f'trying to evaluate {task_name}')

                if datasets_wrapper.eval_method is None:
                    logger.info(f'cannot evaluate dataset {task_name}, because no eval method is provided')

                else:

                    eval_method_name = datasets_wrapper.eval_method
                    predictor_name = datasets_wrapper.predictor
                    logger.info(
                        f'evaluating {task_name} with evaluator: {eval_method_name}, and predictor: {predictor_name}')

                    evaluator = BasicFactory().get_object(eval_method_name)
                    predictor = BasicFactory().get_object(predictor_name)

                    # check if we need to save the task's error distribution in the trainer state
                    task_errors = None
                    if datasets_wrapper.save_error_distribution:
                        task_errors = {}

                    # get dataloader batch size and collator

                    dataloader_batch_size = self.args.per_device_eval_batch_size \
                        if 'batch_size' not in datasets_wrapper.dataloader_args \
                        else datasets_wrapper.dataloader_args['batch_size']

                    if 'no_collator_in_eval' in datasets_wrapper.dataloader_args \
                            and datasets_wrapper.dataloader_args['no_collator_in_eval']:
                        data_collator = None

                    else:
                        data_collator = DynamicInputIdsPaddingDataCollator(self.tokenizer,
                                                                           train_mode=False)
                    # start prediction loop

                    for dataset_id, dataset in enumerate(datasets_wrapper.datasets):

                        if datasets_wrapper.single_dataset:
                            subtask_name = ''
                            subtask_print_name = ''
                        else:
                            subtask_name = f'{datasets_wrapper.datasets_names[dataset_id]}'
                            subtask_print_name = f'_{subtask_name}'
                            logger.info(
                                f'evaluating subtask: {subtask_name}')

                        # metadata fields
                        # the metadata fields are all the string fields that are not question or answer
                        metadata_fields = [k for k, v in dataset[0].items()
                                           if type(v) == str
                                           and k not in ['question', 'answer']]
                        # prediction loop
                        # init variables
                        predictions = []
                        eval_losses = []
                        num_samples = 0

                        # we need to check the collator for the dataset
                        # default is dynamic padding, unless we specifically ask for random (for example, in wikipedia)

                        dataloader = DataLoader(dataset,
                                                batch_size=dataloader_batch_size,
                                                shuffle=True if train_mode else False,
                                                collate_fn=data_collator
                                                )

                        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                             desc='generating predictions for evaluation'):

                            # get prediction
                            input_ids = batch['input_ids'].to(self.args.device)
                            attention_mask = batch['attention_mask'].to(self.args.device)
                            labels = batch['labels'].to(self.args.device)
                            preds = predictor(tokenizer=self.tokenizer,
                                              model=self.model,
                                              input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              labels=labels)

                            # get loss
                            batch_loss = self.prediction_step(self.model,
                                                              inputs={'input_ids': input_ids,
                                                                      'attention_mask': attention_mask,
                                                                      'labels': labels},
                                                              prediction_loss_only=True
                                                              )[0].mean().item()
                            eval_losses += [batch_loss] * len(input_ids)

                            # check if we are doing QA or MLM eval
                            prediction_type = type(preds[0])
                            if prediction_type == SpanPrediction:

                                # for mlm extend the predictions
                                predictions.extend(preds)
                                num_samples += len(preds)

                            else:
                                # a batch can be larger than one
                                for k, pred in enumerate(preds):

                                    # todo how to pass metadata
                                    if 'all_answers' in batch:
                                        gold = json.loads(batch['all_answers'][k])
                                    else:
                                        gold = batch['answer'][k]

                                    # append the merged prediction and metadata dict
                                    prediction_dict = {
                                        'question': batch['question'][k],
                                        'gold': gold,
                                        'prediction': pred}
                                    metadata_dict = {metadata_key: batch[metadata_key][k]
                                                     for metadata_key in metadata_fields}
                                    predictions.append({**prediction_dict, **metadata_dict})
                                    num_samples += 1

                            # todo export to config
                            if train_mode and num_samples >= 1000:
                                break

                        # save predictions
                        dataset_name = f"{task_name}{subtask_print_name}_{'train' if train_mode else 'eval'}"
                        output_predictions_path = os.path.join(self.args.output_dir,
                                                               f"predictions-{self.state.global_step}-{dataset_name}.csv")
                        output_predictions_path_json = os.path.join(self.args.output_dir,
                                                               f"predictions-{self.state.global_step}-{dataset_name}.json")
                        # # save and upload predictions
                        predictions_df = pd.DataFrame(predictions)
                        predictions_df.to_csv(output_predictions_path)

                        # save predictions json for qa tasks
                        if len(predictions) and hasattr(predictions[0], '__iter__') and 'id' in predictions[0]:
                            predictions_json = {q['id']: q['prediction'] for q in predictions}
                            with open(output_predictions_path_json, 'w') as fp:
                                json.dump(predictions_json, fp)

                        # todo upload
                        # wandb.save(output_predictions_path)

                        # evaluate and report predictions
                        result_dict = evaluator().evaluate(predictions=predictions,
                                                           output_predictions_path=output_predictions_path,
                                                           dataset_name=dataset_name)

                        # update result_dict with max for every metric
                        result_dict = self.update_max_eval_metrics(result_dict)

                        # on_predictions_save handler to save predictions to s3/wandb
                        # the evaulator can override the predictions file with eval fields
                        self.qa_callback_handler.on_predictions_save(self.args,
                                                                     self.state,
                                                                     self.control,
                                                                     output_predictions_path,
                                                                     dataset_name)

                        result_dict[f'{dataset_name}_loss'] = np.average(eval_losses)

                        # fill task errors
                        if task_errors is not None:
                            task_errors[subtask_name] = 1 - result_dict[f'{dataset_name}_f1']

                        logger.info(f'Finished evaluating {dataset_name}.')
                        logger.info(f'Results for {dataset_name}: {result_dict}')

                        # report logs to qa handler and hf handler
                        self.callback_handler.on_log(self.args, self.state, self.control,
                                                     result_dict)
                        self.qa_callback_handler.on_log(self.args, self.state, self.control,
                                                        result_dict)

                    # if task errors is not none, save the errors to the state
                    if task_errors is not None and hasattr(self.state, "tasks_errors"):

                        # check if this is the first time we saw the task
                        task_errors_name_for_state = f'{task_name}Errors'
                        if task_errors_name_for_state not in self.state.tasks_errors:
                            self.state.tasks_errors[task_errors_name_for_state] = {}

                        # save the errors with the number of steps as key
                        # if this task has one subtask, save as an int, otherwise save a dict
                        if datasets_wrapper.single_dataset and len(task_errors) == 1:
                            self.state.tasks_errors[task_errors_name_for_state][self.state.global_step] = \
                            list(task_errors.values())[0]
                        else:
                            self.state.tasks_errors[task_errors_name_for_state][self.state.global_step] = task_errors

                    # self.get_train_dataloader()

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """

        # we can either evaluate a dictionary of datasets or a single dataset

        if type(self.eval_dataset) == dict:
            self.evaluate_datasets(self.eval_dataset, train_mode=False)

        if type(self.train_dataset) == dict:
            self.evaluate_datasets(self.train_dataset, train_mode=True)

        # to stop double evaluation on epoch end
        self.control.should_evaluate = False
        self.control.restart_train_dataloader = True

    def restart_sampling_counters(self,
                                  resampling_counter_keys=['']
                                  ):
        """
        iterate over all counter keys, and reset the ones that are relevant for each sampling mini-epoch
        """
        for task_name, task_counter_dict in self.state.task_counter.items():
            for counter_name, counter_value in task_counter_dict.items():
                if counter_name in resampling_counter_keys:
                    self.state.task_counter[task_name][counter_name] = 0

    def train(
            self,
            resume_from_checkpoint: Optional[str] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str`, `optional`):
                Local path to a saved checkpoint as saved by a previous instance of :class:`~transformers.Trainer`. If
                present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # check if we need to use the base train() method from hf or our multi task implementation
        override_huggingface_train_method = Config().get('trainer.override_huggingface_train_method')

        # if override_huggingface_train_method isn't set to true use the HF method
        if override_huggingface_train_method is None or override_huggingface_train_method == False:
            super(BasicTrainer, self).train()

        # else use our implementation
        else:
            if "model_path" in kwargs:
                resume_from_checkpoint = kwargs.pop("model_path")
                warnings.warn(
                    "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                    "instead.",
                    FutureWarning,
                )

            if len(kwargs) > 0:
                raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
            # This might change the seed so needs to run first.
            self._hp_search_setup(trial)

            # Model re-init
            model_reloaded = False
            if self.model_init is not None:
                # Seed must be set before instantiating the model when using model_init.
                set_seed(self.args.seed)
                self.model = self.call_model_init(trial)
                model_reloaded = True
                # Reinitializes optimizer and scheduler
                self.optimizer, self.lr_scheduler = None, None

            # Load potential model checkpoint
            if resume_from_checkpoint is not None and os.path.isfile(
                    os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                logger.info(f"Loading model from {resume_from_checkpoint}).")
                if isinstance(self.model, PreTrainedModel):
                    self.model = self.model.from_pretrained(resume_from_checkpoint)
                    model_reloaded = True
                else:
                    state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
                    self.model.load_state_dict(state_dict)

            # If model was re-initialized, put it on the right device and update self.model_wrapped
            if model_reloaded:
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
                self.model_wrapped = self.model

            # Keeping track whether we can can len() on the dataset or not
            train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

            # Data loader and number of training steps
            train_dataloader = self.get_train_dataloader()

            # Setting up training control variables:
            # number of training epochs: num_train_epochs
            # number of training steps per epoch: num_update_steps_per_epoch
            # total number of training steps to execute: max_steps
            if train_dataset_is_sized:
                num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                if self.args.max_steps > 0:
                    max_steps = self.args.max_steps
                    num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                        self.args.max_steps % num_update_steps_per_epoch > 0
                    )
                else:
                    max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(self.args.num_train_epochs)
            else:
                # see __init__. max_steps is set when the dataset has no __len__
                max_steps = self.args.max_steps
                num_train_epochs = 1
                num_update_steps_per_epoch = max_steps

            if self.args.deepspeed:
                model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
                self.model = model.module
                self.model_wrapped = model  # will get further wrapped in DDP
                self.deepspeed = model  # DeepSpeedEngine object
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
            else:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            self.state = TrainerState()
            self.state.is_hyper_param_search = trial is not None

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

            model = self.model_wrapped

            # Mixed precision training with apex (torch < 1.6)
            if self.use_apex:
                model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

            # Multi-gpu training (should be after apex fp16 initialization)
            if self.args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Distributed training (should be after apex fp16 initialization)
            if self.sharded_dpp:
                model = ShardedDDP(model, self.optimizer)
            elif is_sagemaker_distributed_available():
                model = DDP(model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)
            elif self.deepspeed:
                pass  # already initialized its own DDP earlier
            elif self.args.local_rank != -1:
                if self.args.ddp_find_unused_parameters is not None:
                    find_unused_parameters = self.args.ddp_find_unused_parameters
                elif isinstance(model, PreTrainedModel):
                    # find_unused_parameters breaks checkpointing as per
                    # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                    find_unused_parameters = not getattr(model.config, "gradient_checkpointing", False)
                else:
                    find_unused_parameters = True
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                    find_unused_parameters=find_unused_parameters,
                )

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # important: at this point:
            # self.model         is the Transformers Model
            # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

            # Train!
            if is_torch_tpu_available():
                world_size = xm.xrt_world_size()
            elif self.args.local_rank != -1:
                world_size = dist.get_world_size()
            else:
                world_size = 1

            total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * world_size
            num_examples = (
                self.num_examples(train_dataloader)
                if train_dataset_is_sized
                else total_train_batch_size * self.args.max_steps
            )

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")

            self.state.epoch = 0
            start_time = time.time()
            epochs_trained = 0
            steps_trained_in_current_epoch = 0

            # Check if continuing training from a checkpoint
            if resume_from_checkpoint is not None and os.path.isfile(
                    os.path.join(resume_from_checkpoint, "trainer_state.json")
            ):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
                epochs_trained = self.state.global_step // num_update_steps_per_epoch
                if not self.args.ignore_data_skip:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
                else:
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not self.args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                        "batches in the first epoch."
                    )

            # Update the references
            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader
            self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
            self.state.trial_params = hp_params(trial) if trial is not None else None
            # This should be the same if the state has been saved but in case the training arguments changed, it's safer
            # to set this after the load.
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            # tr_loss is a tensor to avoid synchronization of TPUs through .item()
            tr_loss = torch.tensor(0.0).to(self.args.device)
            # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = self.state.global_step
            self._total_flos = self.state.total_flos
            model.zero_grad()

            self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

            # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
            if not self.args.ignore_data_skip:
                for epoch in range(epochs_trained):
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break

            for epoch in range(epochs_trained, num_train_epochs):
                if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler,
                                                                           DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)

                if is_torch_tpu_available():
                    parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                        self.args.device
                    )
                    epoch_iterator = parallel_loader
                else:
                    epoch_iterator = train_dataloader

                # Reset the past mems state at the beginning of each epoch if necessary.
                if self.args.past_index >= 0:
                    self._past = None

                steps_in_epoch = (
                    len(epoch_iterator)
                    if train_dataset_is_sized
                    else self.args.max_steps * self.args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

                for step, inputs in enumerate(epoch_iterator):

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    # for multi tasking counter
                    self.qa_callback_handler.on_batch_begin(self.args, self.state, self.control, inputs)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        # NEW CODE: call handler on step begin
                        self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                    if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss += self.training_step(model, inputs)
                    else:
                        tr_loss += self.training_step(model, inputs)

                    self._total_flos += self.floating_point_ops(inputs)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            steps_in_epoch <= self.args.gradient_accumulation_steps
                            and (step + 1) == steps_in_epoch
                    ):
                        # Gradient clipping
                        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.use_amp:
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    self.args.max_grad_norm,
                                )

                        # Optimizer step
                        if self.deepspeed:
                            self.deepspeed.step()
                        elif is_torch_tpu_available():
                            xm.optimizer_step(self.optimizer)
                        elif self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                        # report task counter
                        if self.control.report_task_counter:
                            # log how many batches we seen seen for every task
                            logger.info(
                                f'Logging tasks counter: {self.state.task_counter}')

                            # each task has a dict with relevant counters, create keys to log by iterating
                            task_counter_to_log = {}
                            for task_name, task_counter_dict in self.state.task_counter.items():
                                for counter_name, counter_value in task_counter_dict.items():
                                    task_counter_to_log[f'{task_name}_#_{counter_name}'] = counter_value

                            # log relevant task counters, for example
                            self.callback_handler.on_log(self.args,
                                                         self.state,
                                                         self.control,
                                                         task_counter_to_log)
                            self.control.report_task_counter = False

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                        ### --- new code --- START ###

                        # check if we need to replace the train dataloader
                        if self.control.restart_train_dataloader:
                            load_train_dataloader_after_eval = Config().get('trainer.load_train_dataloader_after_eval')
                            if load_train_dataloader_after_eval is not None and load_train_dataloader_after_eval == True:
                                # this is necessary for sub tasks
                                # remove before merge!!!
                                print('updating train dataloader after eval')
                                for task_iterable in train_dataloader.task_iterables.values():
                                    if hasattr(task_iterable['dataloader'], 'update_sampler_with_trainer_state'):
                                        task_iterable['dataloader'].update_sampler_with_trainer_state(self.state)

                                # update the task indices based on the sampler
                                train_dataloader.task_indices = train_dataloader.sampler.sample(
                                    train_dataloader.task_iterables)

                                #train_dataloader = self.get_train_dataloader()

                                epoch_iterator = train_dataloader

                                # restart sampling counters
                                self.restart_sampling_counters(resampling_counter_keys=['Examples_since_resampling',
                                                                                        'Batches_since_resampling'])
                                self.callback_handler.train_dataloader = train_dataloader
                                self.control.restart_train_dataloader = False
                                break

                        ### --- new code --- END ###

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

                self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.args.tpu_metrics_debug or self.args.debug:
                    if is_torch_tpu_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

            if self.args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                logger.info(
                    f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
                )
                if isinstance(self.model, PreTrainedModel):
                    self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                    if not self.is_model_parallel:
                        self.model = self.model.to(self.args.device)
                else:
                    state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                    self.model.load_state_dict(state_dict)

                if self.deepspeed:
                    self.deepspeed.load_checkpoint(
                        self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                    )

            metrics = speed_metrics("train", start_time, self.state.max_steps)
            if self._total_flos is not None:
                self.store_flos()
                metrics["total_flos"] = self.state.total_flos
            self.log(metrics)

            self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()

            return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)
