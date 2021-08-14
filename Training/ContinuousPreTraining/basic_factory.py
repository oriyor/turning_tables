import logging
import os

from ContinuousPreTraining.Common.file_utils import upper_to_lower_notation_name, find_module

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BasicFactory:
    # todo move all other factories here
    """
    factory to get the trainer from the config
    """

    def __init__(self):
        pass

    def find_object(self, object_name):

        object_name_lower = upper_to_lower_notation_name(object_name)
        module_name = find_module(os.path.dirname(os.path.abspath(__file__)), object_name_lower)
        try:
            mod = __import__(f'ContinuousPreTraining.' + module_name,
                             fromlist=[object_name])
        except:
            logger.error(module_name + ' module not found!!')
            assert (ValueError('object not found!'))

        return getattr(mod, object_name)

    def get_object(self, object_name):
        #todo rename
        """
        factory method to get a predictor
        """
        obj = self.find_object(object_name)

        # init predictor and return
        return obj