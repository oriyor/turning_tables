import numpy as np


class RandomSampler:
    """
    indices for a random sample over the iterables
    """
    def sample(self, task_iterables_list):
        task_choice_list = []
        for i, task_iterable in enumerate(task_iterables_list.values()):
            task_choice_list += [i] * task_iterable['num_batches']
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)

        return task_choice_list
