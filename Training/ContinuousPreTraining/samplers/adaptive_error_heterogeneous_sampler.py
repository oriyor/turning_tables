import numpy as np
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


class AdaptiveErrorHeterogeneousSampler:
    """
    indices for a uniform sample based on the iterable size between iterable
    """

    def __init__(self,
                 trainer_state,
                 is_adaptive,
                 normalize_with_prev,
                 distribution_name
                 ):
        """
        init the dataloaders for each task
        get the sampling indices between the tasks
        """
        self.is_adaptive = is_adaptive
        self.trainer_state = trainer_state
        self.last_adaptive_weights = None
        self.normalize_with_prev = normalize_with_prev
        self.distribution_name = distribution_name

    def update_sampler_trainer_state(self, trainer_state):
        """
        update trainer state after calculating errors
        """
        self.trainer_state = trainer_state

    def sample(self, task_iterables_list):

        num_tasks = len(task_iterables_list)

        #if self.last_adaptive_weights is None:
        #    self.last_adaptive_weights = [1 / num_tasks for i in range(num_tasks)]

        task_errors_name_for_state = f'{self.distribution_name}Errors'

        # try and retrieve the error distribution from the trainer state
        if hasattr(self.trainer_state, 'tasks_errors') and \
                len(list(self.trainer_state.tasks_errors.values())[0]) > 1 and self.is_adaptive:

            # reverse the dict
            errors_dict = {}
            for step_num, tasks_name_values in self.trainer_state.tasks_errors[task_errors_name_for_state].items():
                for task_name, task_error_value in tasks_name_values.items():
                    if task_name not in errors_dict:
                        errors_dict[task_name] = {}
                    errors_dict[task_name][step_num] = task_error_value

            # iterate subtask and find error for each task
            delta_errors = []
            for i, sub_task in enumerate(task_iterables_list.keys()):

                # look for each subtask in the error distribution
                if sub_task not in errors_dict:
                    raise Exception(f"Subtask {sub_task} not found in error distribution")
                else:
                    error_list = list(errors_dict[sub_task].values())

                    #if self.normalize_with_prev:
                    #    task_weight = max(0.01, error_list[-1] - error_list[-2]) / self.last_adaptive_weights[i]
                    #else:
                    window = error_list[-min(len(error_list), 4):]
                    window_gains = ((window[-1] + window[-2]) - (window[1] + window[0])) / len(window)
                    task_weight = max(0.002, abs(window_gains))
                    delta_errors.append(task_weight)

            # normalize errors
            adaptive_weights = np.array(delta_errors) / sum(delta_errors)

        # else, init the uniform distribution
        else:
            adaptive_weights = [1 / num_tasks for i in range(num_tasks)]

        print('Error Distributions:')
        print({k: adaptive_weights[i]
               for i, k in enumerate(task_iterables_list.keys())})
        return adaptive_weights
