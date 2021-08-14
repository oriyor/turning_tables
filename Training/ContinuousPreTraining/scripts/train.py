import os

# set wandb api key before imports
#from ContinuousPreTraining.Data.updated_dr_factory import get_multi_task_datasets, get_multi_task_dataset
from ContinuousPreTraining.Data.updated_dr_factory import DatasetReaderFactory

from ContinuousPreTraining.Common.transfomer_utils import get_tokenizer, get_model
from ContinuousPreTraining.Training.callback_factory import CallbackFactory
from ContinuousPreTraining.Training.optimizers import get_optimizer, get_scheduler
from ContinuousPreTraining.Common.config import Config

from transformers import TrainingArguments, Trainer
import argparse
import logging
import sys
import jsoncfg
import math
import wandb

from ContinuousPreTraining.Training.trainer_factory import TrainerFactory

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


def get_training_args(training_args_config, args, train_dataset):
    # get params such that cmd args run over the config
    num_epochs = Config().get('training_arguments.num_train_epochs')
    per_device_train_batch_size = Config().get('training_arguments.per_device_train_batch_size')
    per_device_eval_batch_size = Config().get('training_arguments.per_device_eval_batch_size')
    gradient_accumulation_steps = Config().get('training_arguments.gradient_accumulation_steps')
    log_steps = Config().get('training_arguments.log_steps')

    # take eval and save steps from config
    save_steps = int(Config().get('training_arguments.save_steps'))
    eval_steps = Config().get('training_arguments.eval_steps')
    if eval_steps is not None:
        eval_steps = int(eval_steps)

    # take evaluation strategy from config
    evaluation_strategy = Config().get('training_arguments.evaluation_strategy')
    if evaluation_strategy is None:
        evaluation_strategy = 'steps'

    # mainly for debugging only on cpu
    no_cuda = Config().get('training_arguments.no_cuda')
    if no_cuda is None:
        no_cuda = False

    return TrainingArguments(
        output_dir=Config().get('experiment.experiment_name') + '_' + Config()._start_time,  # output directory
        num_train_epochs=num_epochs,  # total # of training epochs
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,
        prediction_loss_only=training_args_config['prediction_loss_only'],
        weight_decay=training_args_config['weight_decay'],  # strength of weight decay
        logging_dir='logs',  # directory for storing logs
        evaluation_strategy=evaluation_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=log_steps,
        save_total_limit=training_args_config['save_total_limit'],
        logging_first_step=True,
        seed=int(Config().get('training_arguments.seed')),
        no_cuda=no_cuda
    )


def main(args):
    # get config
    Config().load(args.config_path)
    Config().override_dict(vars(args))
    config = jsoncfg.load_config(args.config_path)
    use_wandb = args.use_wandb
    path_prefix = os.getcwd()

    # get model and tokenizer
    logger.info('Initializing model and tokenizer')

    # override model or tokenizer from commandline
    model_config = Config().get('model')

    tokenizer_name = config.tokenizer() \
        if args.tokenizer is None else args.tokenizer

    # get the model and tokenizer
    model = get_model(model_config)
    tokenizer = get_tokenizer(tokenizer_name)

    logger.info('Initializing train and validation datasets')
    # get training dataset: for this we need the path prefix to the data, the tokenizer, and the config
    # train_dataset = DatasetReaderFactory().get_dataset_reader(path_prefix, tokenizer, Config().get('train_dataset_reader'))
    logger.info('Initializing train datasets')
    train_dataset = DatasetReaderFactory().get_multi_task_dataset(path_prefix,
                                                                  tokenizer,
                                                                  Config().get('train_datasets'),
                                                                  is_train_task=True)

    # get validation dataset: for this we need the path prefix to the data, the tokenizer, and the config
    logger.info('Initializing validation datasets')
    validation_dataset = DatasetReaderFactory().get_multi_task_dataset(path_prefix,
                                                                       tokenizer,
                                                                       Config().get('validation_datasets'),
                                                                       is_train_task=False)

    # validation_dataset = DatasetReaderFactory().get_dataset_reader(path_prefix, tokenizer,
    # Config().get('validation_dataset_reader'))

    logger.info('Initializing optimizer and scheduler')
    # get optimizer and scheduler
    optimizer = get_optimizer(config.optimizer(), model, Config().get('optimizer.lr'))
    scheduler = get_scheduler(optimizer, config.scheduler())

    logger.info('Training args and trainer')
    # get training args
    training_args = get_training_args(config.training_arguments(), args, train_dataset)

    # get additional callbacks for the trainer
    trainer_config = config.trainer()
    callbacks = []
    if 'callbacks' in trainer_config:
        for callback in trainer_config['callbacks']:
            callbacks.append(CallbackFactory().get_callback(callback))

    # get trainer
    trainer_args = {'model': model,
                    'args': training_args,
                    'train_dataset': train_dataset,
                    'eval_dataset': validation_dataset,
                    'optimizers': (optimizer, scheduler),
                    'callbacks': callbacks,
                    'use_wandb': use_wandb
                    }

    trainer = TrainerFactory().get_trainer(trainer_config, trainer_args, tokenizer)

    # # train
    # if use_wandb:
    #     wandb.log(training_args.to_dict())
    #     wandb.log(model.config.to_dict())

    # check if we only want to do evaluation
    # if evaluation_only is chosen, we will do eval instead of train
    evaluation_only = Config().get('trainer.evaluation_only')
    if evaluation_only is not None and evaluation_only:
        trainer.evaluate()

    # else, we want to train the model
    else:
        trainer.train()

    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process hyperparameters.')
    parser.add_argument('-c', '--config_path', type=str, default='../configurations/lot_train_config.json',
                        help='path to config file')
    parser.add_argument('-t', '--tokenizer', type=str, default=None,
                        help='override tokenizer name')
    parser.add_argument('-lr', '--optimizer.lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('-e', '--training_arguments.num_train_epochs', type=int, default=None,
                        help='number of epochs')
    parser.add_argument('-gas', '--training_arguments.gradient_accumulation_steps', type=int, default=None,
                        help='gradient accumulation steps')
    parser.add_argument('-tbs', '--training_arguments.per_device_train_batch_size', type=int, default=None,
                        help='train batch size')
    parser.add_argument('-ebs', '--training_arguments.per_device_eval_batch_size', type=int, default=None,
                        help='train batch size')
    parser.add_argument('-ls', '--training_arguments.log_steps', type=int, default=100,
                        help='log every number of steps')
    parser.add_argument("-w", '--use_wandb', action='store_true', help="if added, wandb is used", default=False)

    # supporting unknown arguments
    args, unknown_args = parser.parse_known_args()
    for argument in unknown_args:
        if argument.startswith('-'):
            parser.add_argument(argument.strip())
    args, unknown_args = parser.parse_known_args()
    main(args)
