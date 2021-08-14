from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup


def get_optimizer(optimizer_config, model, args_lr):
    """
    helper method to get an optimizer
    """
    lr = optimizer_config['lr'] if args_lr is None else args_lr

    # traverse the possible optimizers
    if optimizer_config['type'] == 'AdaFactor':
        return Adafactor(  # https://discuss.huggingface.co/t/t5-finetuning-tips/684/3
            model.parameters(),
            lr=lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

    if optimizer_config['type'] == 'AdamW':
        return AdamW(model.parameters(), lr=lr)


def get_scheduler(optimizer, scheduler_config):
    """
    helper method to get a scheduler
    """
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=scheduler_config['num_warmup_steps'],
        num_training_steps=scheduler_config['num_training_steps']
    )
