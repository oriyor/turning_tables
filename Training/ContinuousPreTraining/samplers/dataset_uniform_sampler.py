import numpy as np


class DatasetUniformSampler:
    """
    indices for a uniform sample based on the iterable size between iterable
    """

    def sample(self, task_iterables_list):
        """
        sample indices
        """
        max_num_batches_for_task = max([task_iterable['num_batches']
                                        for task_iterable in task_iterables_list.values()])
        task_choice_list = []
        for i, task_iterable in enumerate(task_iterables_list.values()):
            task_choice_list += [i] * max_num_batches_for_task
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)

        return task_choice_list

