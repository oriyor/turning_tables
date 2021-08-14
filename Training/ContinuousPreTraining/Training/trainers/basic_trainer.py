from typing import Optional, Dict

import os
from datasets import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import Trainer
from ContinuousPreTraining.Training.callback_factory import CallbackFactory
from ContinuousPreTraining.Training.trainer_callbacks.basic_qa_callback_handler import BasicQaCallbackHandler


class BasicTrainer(Trainer):
    """
    trainer that can upload checkpoints to WandB
    """

    def __init__(self, **kwargs):
        """
        we add a new field that states whether we use in wandb
        """
        self.use_wandb = kwargs['use_wandb']
        del kwargs['use_wandb']
        super().__init__(**kwargs)

        # add saving callbacks
        saving_callbacks = []
        for saving_callback_name in []:
            saving_callbacks.append(CallbackFactory().get_callback(saving_callback_name))

        self.qa_callback_handler = BasicQaCallbackHandler(list(set(kwargs['callbacks'] + saving_callbacks)),
                                                          self.model,
                                                          self.tokenizer,
                                                          self.optimizer,
                                                          self.lr_scheduler)


    def _save_checkpoint(self, model, trial, metrics=None):
        # call original save checkpoint
        super()._save_checkpoint(model, trial, metrics=metrics)
        self.qa_callback_handler.on_save(self.args, self.state, self.control,
                                         prefix_checkpoint=PREFIX_CHECKPOINT_DIR,
                                         use_wandb=self.use_wandb)

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
        super().evaluate(eval_dataset)

        # we also pass the tokenizer and eval dataset
        self.qa_callback_handler.on_evaluate(self.args, self.state, self.control,
                                             tokenizer=self.tokenizer,
                                             eval_dataset=self.eval_dataset,
                                             train_dataset=self.train_dataset,
                                             use_wandb=self.use_wandb)
