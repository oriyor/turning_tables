import logging
from ContinuousPreTraining.Training.trainer_callbacks.basic_qa_callback import BasicQaCallback

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MultiTaskHeterogeneousCallback(BasicQaCallback):
    """
    A :class:`~transformers.TrainerCallback` for handling multi task events
    """

    def __init__(self):
        self._initialized = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # init multi task specific fields
        state.task_counter = {}
        state.tasks_indices = {}
        state.tasks_errors = {}

        control.restart_train_dataloader = False
        control.report_task_counter = False


    def on_batch_begin(self, args, state, control, model=None, **kwargs):
        """
        for every step, update the counter for the current step
        """
        if 'batch_inputs' in kwargs:
            batch_inputs = kwargs['batch_inputs']

            if 'batches' not in state.task_counter:
                state.task_counter['batches'] = {'without_subtask': 0,
                                                 'with_subtask': 0}

            # update num of batches withput subtask names
            if type(batch_inputs['task_name']) == str:
                # update the counter for the task
                state.task_counter['batches']['without_subtask'] += 1
                train_task_name = batch_inputs['task_name']
                del batch_inputs['task_name']

                if train_task_name not in state.task_counter:
                    state.task_counter[train_task_name] = {}
                    state.task_counter[train_task_name]['Batches_total'] = 0
                    state.task_counter[train_task_name]['Batches_since_resampling'] = 0
                    state.task_counter[train_task_name]['Examples'] = 0

                # update the batches and examples counter
                state.task_counter[train_task_name]['Batches_total'] += 1
                state.task_counter[train_task_name]['Batches_since_resampling'] += 1
                state.task_counter[train_task_name]['Examples'] += len(batch_inputs['input_ids'])

            # else update the number of examples for each sub_task
            else:

                state.task_counter['batches']['with_subtask'] += 1
                # get batch task name and delete
                train_task_name = batch_inputs['task_name']
                del batch_inputs['task_name']

                # update the index dict with the tasks name
                state.tasks_indices[state.global_step] = train_task_name

                # update the counter for the task
                for task_name, task_example_count in dict(train_task_name).items():
                    if task_name not in state.task_counter:
                        state.task_counter[task_name] = {}
                        state.task_counter[task_name]['Examples_total'] = 0
                        state.task_counter[task_name]['Examples_since_resampling'] = 0
                    # update the batches and examples counter
                    state.task_counter[task_name]['Examples_total'] += task_example_count
                    state.task_counter[task_name]['Examples_since_resampling'] += task_example_count

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        after every step, check if we need to report the task counter
        """

        if control.should_evaluate:
            control.report_task_counter = True

