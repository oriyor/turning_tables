import logging
import os
from ContinuousPreTraining.Common.file_utils import upper_to_lower_notation_name, find_module

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CallbackFactory:
    """
    factory to get the trainer from the config
    """

    def __init__(self):
        pass

    def find_callback(self, callback_name):

        callback_name_lower = upper_to_lower_notation_name(callback_name)
        module_name = find_module(os.path.dirname(os.path.abspath(__file__)), callback_name_lower)
        try:
            mod = __import__('ContinuousPreTraining.Training.' + module_name,
                             fromlist=[callback_name])
        except:
            logger.error(module_name + ' module not found!!')
            assert (ValueError('qgen_name not found!'))

        return getattr(mod, callback_name)

    def get_callback(self, callback_name):
        """
        factory method to get a trainer
        """
        callback = self.find_callback(callback_name)

        # init trainer and return
        return callback
