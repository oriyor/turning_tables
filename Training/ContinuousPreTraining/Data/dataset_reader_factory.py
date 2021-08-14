import os

from ContinuousPreTraining.Common.file_utils import upper_to_lower_notation_name, find_module, cached_path
from ContinuousPreTraining.Data.data_utils import build_data_blocks

import logging
import pandas as pd


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DatasetReaderFactory:
    """
    factory to get a dataset from config
    """

    def __init__(self):
        pass

    def find_dataset_reader(self, dataset_reader_name):

        dataset_reader_lower = upper_to_lower_notation_name(dataset_reader_name)
        module_name = find_module(os.path.dirname(os.path.abspath(__file__)), dataset_reader_lower)
        try:
            mod = __import__('ContinuousPreTraining.Data.' + module_name,
                             fromlist=[dataset_reader_name])
        except:
            logger.error(module_name + ' module not found!!')
            assert (ValueError('qgen_name not found!'))

        return getattr(mod, dataset_reader_name)

    def get_multi_task_dataset(self, path_prefix, tokenizer, ds_config):
        """
        get a multi task dataset from a config
        """
        # we will return a dictionary between datasets names and their corresponding datasets
        multi_task_dataset = {}

        for dataset_name, dataset_config in ds_config['datasets'].items():

            # whether we are split the dataset to blocks of similar size
            if 'block_size' in dataset_config:
                dataset_input_path = dataset_config['path']
                # supporting http data
                if dataset_input_path.startswith('http'):
                    dataset_file_path = cached_path(dataset_input_path)
                else:
                    dataset_file_path = f'{path_prefix}{dataset_input_path}'

                sorted_lengsth_data_blocks = build_data_blocks(dataset_file_path
                    , dataset_config['block_size'])

                # traverse all blocks
                dataset_reader = self.find_dataset_reader(dataset_config['type'])

                # config
                source_len = dataset_config['max_seq_len']
                generation_model = dataset_config['generation_model']
                output_max_len = dataset_config['output_max_len']

                # traverse all blocks
                for i, block in enumerate(sorted_lengsth_data_blocks):
                    if 'phrase' in block.block[0]:
                        multi_task_dataset[f'{dataset_name}_{i}'] = \
                            {
                                'dataset': dataset_reader(
                                    pd.DataFrame([[example['phrase'],
                                                   example['context'],
                                                   example['answer']]
                                                  for example in block.block], columns=['phrases', 'contexts', 'gold']),
                                    tokenizer=tokenizer,
                                    source_len=source_len,
                                    generation_model=generation_model,
                                    output_max_len=output_max_len
                                ),
                                'batch_size_ratio': 1, # this is the main dataset, we will keep the regular batch ratio
                                'collator': 'smart'  # we use smart collator when we split to blocks
                            }
                    else:
                        multi_task_dataset[f'{dataset_name}_{i}'] = \
                            {
                                'dataset': dataset_reader(
                                    pd.DataFrame([[example['context'],
                                                   example['answer']]
                                                  for example in block.block], columns=['contexts', 'gold']),
                                    tokenizer=tokenizer,
                                    max_seq_len=dataset_config['max_seq_len'],
                                    generation_model=dataset_config['generation_model']),
                                'batch_size_ratio': 1,  # this is the main dataset, we will keep the regular batch ratio
                                'collator': 'smart'  # we use smart collator when we split to blocks
                            }

            else:
                # get params for multi task dataset
                batch_size_ratio = dataset_config['batch_size_ratio']
                collator = dataset_config['collator']
                del dataset_config['batch_size_ratio']
                del dataset_config['collator']

                # add the dataset to the dictionary
                multi_task_dataset[dataset_name] = {'dataset': self.get_single_task_dataset(path_prefix,
                                                                                tokenizer,
                                                                                dataset_config),
                                                    'batch_size_ratio': batch_size_ratio,
                                                    'collator': collator
                                                    }

        return multi_task_dataset

    def get_single_task_dataset(self, path_prefix, tokenizer, ds_config):
        """
        get a single task dataset from a config
        """
        dataset_reader = self.find_dataset_reader(ds_config['type'])
        dataset_input_path = ds_config['path']

        # add args to config
        # supporting http data
        if dataset_input_path.startswith('http'):
            dataset_file_path = cached_path(dataset_input_path)
        else:
            dataset_file_path = f'{path_prefix}{dataset_input_path}'
        ds_config['data_path'] = dataset_file_path
        if ds_config['pass_tokenizer']:
            ds_config['tokenizer'] = tokenizer

        # delete args from config
        del ds_config['path']
        del ds_config['pass_tokenizer']
        del ds_config['type']

        ds_args = ds_config
        dataset = dataset_reader(**ds_args)
        return dataset

    def get_dataset_reader(self, path_prefix, tokenizer, ds_config):
        ds_type = ds_config['type']
        if ds_type == 'multi_task':
            dataset = self.get_multi_task_dataset(path_prefix, tokenizer, ds_config)
        else:
            dataset = self.get_single_task_dataset(path_prefix, tokenizer, ds_config)
        return dataset
