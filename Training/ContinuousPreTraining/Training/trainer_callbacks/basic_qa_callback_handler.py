from transformers import TrainingArguments
from transformers.trainer_callback import CallbackHandler, TrainerState, TrainerControl
import logging

logger = logging.getLogger(__name__)

class BasicQaCallbackHandler(CallbackHandler):
    """
    callback handler for qa events
    """

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                prefix_checkpoint, use_wandb):
        return self.call_event("on_save", args, state, control,
                               prefix_checkpoint=prefix_checkpoint,
                               use_wandb=use_wandb)

    def on_batch_begin(self, args: TrainingArguments,
                      state: TrainerState,
                      control: TrainerControl,
                      batch_inputs):
        """
        in on_step_begin we want to pass the batch inputs to the event handler so we can update the counter for each task
        """
        self.call_event("on_batch_begin", args, state, control,
                        batch_inputs=batch_inputs)

    def on_predictions_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                            output_predictions_path,
                            dataset_name):
        return self.call_event("on_predictions_save", args, state, control,
                               output_predictions_path=output_predictions_path,
                               dataset_name=dataset_name)
