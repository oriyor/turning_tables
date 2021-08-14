from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class BasicQaCallback(TrainerCallback):
    """
    a basic qa callback class, that implements on_save_predictions call
    """

    def on_predictions_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the :class:`~transformers.Trainer`.
        """
        pass

    def on_batch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of each training batch, as opposed to training step (to running multi-task gas)
        """
        pass