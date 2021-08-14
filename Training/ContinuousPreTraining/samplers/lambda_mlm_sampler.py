import numpy as np


class LambdaMlmSampler:
    """
    task at index 0 is seen lambda percent of the time, the other task 1-lambda
    """

    def __init__(self,
                 lmbda
                 ):
        """
        init the dataloaders for each task
        get the sampling indices between the tasks
        """
        self.lmbda = float(lmbda)

    def sample(self, task_iterables_list):
        """
        sample indices
        """
        # we must have exactly two tasks
        assert len(task_iterables_list) == 2

        # the name of the first task must be WikiTrain
        assert [k for k in task_iterables_list][0] == 'WikiTrain'

        # get the number of batches for the second task, add lambda batches for mlm
        num_batches_for_second_task = list(task_iterables_list.values())[1]['num_batches']
        alpha = (1-self.lmbda)/self.lmbda
        task_choice_list = [0]*int(num_batches_for_second_task*alpha) + [1]*num_batches_for_second_task
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)

        return task_choice_list

