class DatasetsWrapper:
    """
    class to wrap a list of datasets, that share the same config
    this will include a list of datasets, the sampler between the datasets, the args for the dataloader, etc..
    """

    def __init__(self,
                 datasets,
                 dataloader_args,
                 datasets_names = None,
                 sampler='random',
                 predictor=None,
                 eval_method=None,
                 save_error_distribution=None,
                 is_train_task=True
                 ):

        self.datasets = datasets
        self.datasets_names = datasets_names
        self.sampler = sampler
        self.dataloader_args = dataloader_args
        self.single_dataset = True if len(datasets) == 1 else False
        self.num_examples = sum([len(dataset) for dataset in self.datasets])
        self.predictor = predictor
        self.eval_method = eval_method
        self.save_error_distribution = save_error_distribution
        self.is_train_task = is_train_task

