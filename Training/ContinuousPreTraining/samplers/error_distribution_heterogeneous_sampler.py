import numpy as np
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


class ErrorDistributionHeterogeneousSampler:
    """
    indices for a uniform sample based on the iterable size between iterable
    """

    def __init__(self,
                 distribution_name,
                 trainer_state,
                 temperature=1
                 ):
        """
        init the dataloaders for each task
        get the sampling indices between the tasks
        """
        self.distribution_name = distribution_name
        self.trainer_state = trainer_state
        self.temperature = float(temperature)

    def update_sampler_trainer_state(self, trainer_state):
        """
        update trainer state after calculating errors
        """
        self.trainer_state = trainer_state

    def sample(self, task_iterables_list):

        num_tasks = len(task_iterables_list)

        # try and retrieve the error distribution from the trainer state
        task_errors_name_for_state = f'{self.distribution_name}Errors'
        if hasattr(self.trainer_state,
                   'tasks_errors') and task_errors_name_for_state in self.trainer_state.tasks_errors:

            # get the last validation errors for error distribution sampling
            validation_errors = self.trainer_state.tasks_errors[task_errors_name_for_state]
            last_validation_errors_key = max(validation_errors.keys())
            last_validation_errors = validation_errors[last_validation_errors_key]

            # iterate subtask and find error for each task
            errors = []
            for sub_task in task_iterables_list.keys():

                # look for each subtask in the error distribution
                if sub_task not in last_validation_errors:
                    raise Exception(f"Subtask {sub_task} not found in error distribution")
                else:
                    errors.append(last_validation_errors[sub_task])

            # normalize errors and use temperature
            errors = np.power(errors, self.temperature)
            error_distribution = np.array(errors) / sum(errors)

        # else, init the uniform distribution
        else:
            error_distribution = [1 / num_tasks for i in range(num_tasks)]

        print('Error Distributions Heterogeneous:')
        print({k: error_distribution[i]
               for i, k in enumerate(task_iterables_list.keys())})

        return error_distribution
