import copy
from ContinuousPreTraining.Data.datasets_wrapper import DatasetsWrapper
import os
from ContinuousPreTraining.Common.file_utils import upper_to_lower_notation_name, find_module, cached_path
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DatasetReaderFactory:
    """
    factory to get a dataset from config
    """

    def __init__(self):
        self.datasets_to_explode = ['SyntheticQuestionsMultiDatasets']

    def find_dataset_reader(self, dataset_reader_name):

        dataset_reader_lower = upper_to_lower_notation_name(dataset_reader_name)
        module_name = find_module(os.path.dirname(os.path.abspath(__file__)), dataset_reader_lower)
        try:
            mod = __import__('ContinuousPreTraining.Data.' + module_name,
                             fromlist=[dataset_reader_name])
        except:
            logger.error(module_name + ' module not found!!')
            assert (ValueError('datareader name not found!'))

        return getattr(mod, dataset_reader_name)


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

        # add train datasets flag to config
        ds_args = ds_config
        dataset = dataset_reader(**ds_args)
        return dataset

    def init_and_explode_dataset_from_config(self, path_prefix, tokenizer, ds_config):
        """
        input: dataset reader config
        output: will return a list of datasets based on the reader config
        """

        # if the dataset is a single dataset, similar to get_single_task_dataset()
        # else explode and return a list of datasets
        datasets = []
        dataset_reader = self.find_dataset_reader(ds_config['type'])
        dataset_input_path = ds_config['path']
        # supporting http data
        #if dataset_input_path.startswith('http'):
            # todo list files from s3
            # directory_files = [
            #                     # 'counting',
            #                     # 'numeric_superlatives',
            #                     # 'numeric comparison',
            #                     # 'composition_2_hop',
            #                     #'composition',
            #                     # 'numeric_comparison_boolean',
            #                     # 'temporal_comparison',
            #                     # 'temporal_difference',
            #                     # 'temporal_comparison_boolean',
            #                     # 'conjunction',
            #                     # 'arithmetic_superlatives',
            #                     #'arithmetic_addition',
            #                      'most_quantifier',
            #                     # 'only_quantifier',
            #                     # 'every_quantifier',
            #                      'temporal_superlatives'
            #                 ]

        if "skills" in ds_config:
            # get files from s3
            directory_files = ds_config['skills']
            del ds_config['skills']


            for explode_file in directory_files:
                curr_ds_config = copy.deepcopy(ds_config)
                curr_ds_config['path'] = f'{dataset_input_path}{explode_file}.gz'
                dataset = self.get_single_task_dataset(path_prefix, tokenizer, curr_ds_config)
                datasets.append(dataset)

        return datasets, directory_files

    def get_dataset_reader(self, path_prefix, tokenizer, ds_config):
        ds_type = ds_config['type']
        if ds_type == 'multi_task':
            dataset = self.get_multi_task_dataset(path_prefix, tokenizer, ds_config)
        else:
            dataset = self.get_single_task_dataset(path_prefix, tokenizer, ds_config)
        return dataset

    def get_multi_task_dataset(self, path_prefix,
                               tokenizer,
                               datasets_config,
                               is_train_task):
        """
        input: datasets config
        returns: a dictionary between the dataset name and DatasetsWrapper for every dataset in the config
        """
        multi_task_dataset_dict = {}

        for dataset_name, dataset_config in datasets_config.items():

            logger.info(f'Reading data for task: {dataset_name}')
            # init the a list of datasets for every config instance
            if dataset_config['reader']['type'] in self.datasets_to_explode:
                # todo add an if for init/ explode
                initialized_datasets, datasets_names = self.init_and_explode_dataset_from_config(path_prefix,
                                                                                tokenizer,
                                                                                dataset_config['reader'])

            else:
                # add the datasets wrapper to the dictionary
                initialized_datasets = [self.get_single_task_dataset(path_prefix,
                                                                      tokenizer,
                                                                     dataset_config['reader'])]
                datasets_names = None

            # in bot cases, init a datasets wrapper
            multi_task_dataset_dict[dataset_name] = DatasetsWrapper(datasets=initialized_datasets,
                                                                    dataloader_args=dataset_config['dataloader'],
                                                                    datasets_names = datasets_names,
                                                                    sampler=dataset_config.get('dataset_sampler'),
                                                                    predictor=dataset_config.get('predictor'),
                                                                    eval_method=dataset_config.get('eval_method'),
                                                                    save_error_distribution=dataset_config.get('save_error_distribution'),
                                                                    is_train_task=is_train_task
                                                                    )
        return multi_task_dataset_dict


