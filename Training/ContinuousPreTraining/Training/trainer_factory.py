import logging
import os
from transformers import default_data_collator
from ContinuousPreTraining.Common.file_utils import upper_to_lower_notation_name, find_module

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainerFactory:
    """
    factory to get the trainer from the config
    """
    def __init__(self):
        pass


    def find_trainer(self, trainer_name):

        trainer_name_lower = upper_to_lower_notation_name(trainer_name)
        module_name = find_module(os.path.dirname(os.path.abspath(__file__)), trainer_name_lower)
        try:
            mod = __import__('ContinuousPreTraining.Training.' + module_name,
                             fromlist=[trainer_name])
        except:
            logger.error(module_name + ' module not found!!')
            assert (ValueError('qgen_name not found!'))

        return getattr(mod, trainer_name)

    def get_trainer(self, trainer_config, trainer_args, tokenizer):
        """
        factory method to get a trainer
        """
        trainer_name = trainer_config['type']
        trainer = self.find_trainer(trainer_name)

        # check if we need to init a data collator
        if 'data_collator' in trainer_config:
            if trainer_config['data_collator'] == 'smart':
                trainer_args['data_collator'] = SmartDataCollator(tokenizer)
        else:
            trainer_args['data_collator'] = default_data_collator

        # add the tokenizer, as we may need it for eval
        trainer_args['tokenizer'] = tokenizer

        # init trainer and return
        return trainer(**trainer_args)

