import logging
import transformers
import os
from ContinuousPreTraining.Common.file_utils import  s3_get
from transformers import RobertaTokenizer, BertTokenizer, AutoTokenizer, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
transformers.logging.set_verbosity_info()

def get_tokenizer(tokenizer_name, lower_case=True):
    """
    :param tokenizer_name: named identifier for the tokenizer
    :param lower_case: whether we want lower case tokenizer
    :return: tokenizer from hf
    """
    if 't5' in tokenizer_name:
        return T5Tokenizer.from_pretrained(tokenizer_name)
    if tokenizer_name == 'roberta-base':
        return RobertaTokenizer.from_pretrained(tokenizer_name, do_lower_case=lower_case)
    if tokenizer_name == 'bert-base-uncased':
        return BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=lower_case)
    if tokenizer_name == 'bert-large-uncased-whole-word-masking':
        return AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer_name == 'bart-base':
        return BartTokenizer.from_pretrained('facebook/bart-base')
    return None


def get_model(model_config, local_directory=None):
    """
    :param model_name: named identifier for the model
    :param local_directory: whether to start the model from a local directory
    :return: model from hf
    """
    if model_config['size'] == 'Base':
        model_name = 't5-base'
    else:
        if model_config['size'] == 'Large':
            model_name = 't5-large'
        else:
            assert False

    # check if we need to restore the model
    if model_config['PReasM']:
        sampler = model_config['sampler']
        size = model_config['size']

        # create a local dir for the model
        local_dir = f'CheckpointsRestored/PReasM-{sampler}-{size}/'
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        # get model checkpoint files from s3
        s3_directory_url = f's3://tabreas/PReasM/PReasM-{sampler}-{size}/'
        for file_to_restore in ["config.json", "pytorch_model.bin", "optimizer.pt",
                                "scheduler.pt", "trainer_state.json", "training_args.bin"]:

            # download a file from s3
            local_path = local_dir + file_to_restore
            s3_path = s3_directory_url + file_to_restore

            logger.info(f'Downloading {file_to_restore} to {local_path}')

            with open(local_path, "wb") as f:
                s3_get(s3_path, f)

        logger.info(f'Downloaded checkpoint to {local_dir}')

    # get the model
    if 't5' in model_name:
        if model_config['PReasM']:
            return T5ForConditionalGeneration.from_pretrained(local_dir, return_dict=True)
        else:
            logger.info(f'Getting model from huggingface')
            return T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

    else:
        assert False
