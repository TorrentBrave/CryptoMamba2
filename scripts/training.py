import os
import sys
import pathlib
import warnings
from argparse import ArgumentParser

import yaml
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from utils import io_tools
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_root():
    return io_tools.get_root(__file__, num_returns=2)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, help="Logging directory.")
    parser.add_argument("--accelerator", type=str, default='gpu', help="The type of accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of computing devices.")
    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument("--expname", type=str, default='Cmamba', help="Experiment name.")
    parser.add_argument("--config", type=str, default='cmamba_nv', help="Config file name.")
    parser.add_argument("--logger_type", default='tb', type=str, help="Logger type.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument('--save_checkpoints', default=False, action='store_true')
    parser.add_argument('--use_volume', default=False, action='store_true')
    parser.add_argument('--resume_from_checkpoint', default=None)
    parser.add_argument('--max_epochs', type=int, default=200)
    return parser.parse_args()

def save_hparams(log_dir, args):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'hparams.yaml')
    if not os.path.exists(path):
        with open(path, 'w') as f:
            yaml.dump(vars(args), f)

def load_config(root, config_name):
    config_path = os.path.join(root, f'configs/training/{config_name}.yaml')
    return io_tools.load_config_from_yaml(config_path)

def load_data_config(root, data_config_name):
    data_config_path = os.path.join(root, f"configs/data_configs/{data_config_name}.yaml")
    return io_tools.load_config_from_yaml(data_config_path)

def build_transforms(config, use_volume):
    features = config.get('additional_features', [])
    return (
        DataTransform(is_train=True, use_volume=use_volume, additional_features=features),
        DataTransform(is_train=False, use_volume=use_volume, additional_features=features),
        DataTransform(is_train=False, use_volume=use_volume, additional_features=features)
    )

def load_model(config, logger_type, root):
    arch_config = io_tools.load_config_from_yaml(os.path.join(root, 'configs/models/archs.yaml'))
    model_arch = config.get('model')
    model_config_path = os.path.join(root, f'configs/models/{arch_config.get(model_arch)}')
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get('normalize', False)
    hyperparams = config.get('hyperparams')
    if hyperparams:
        model_config['params'].update(hyperparams)
    model_config['params']['logger_type'] = logger_type
    model = io_tools.instantiate_from_config(model_config)
    model.cuda()
    model.train()
    return model, normalize

def get_logger(logger_type, name, args, config):
    if logger_type == 'tb':
        logger = TensorBoardLogger("logs", name=name)
        logger.log_hyperparams(args)
    elif logger_type == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.expname, config={**vars(args), **config})
    else:
        raise ValueError('Unknown logger type.')
    return logger

def get_callbacks(save_checkpoints):
    callbacks = []
    if save_checkpoints:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                save_top_k=1,
                verbose=True,
                monitor="val/rmse",
                mode="min",
                filename='epoch{epoch}-val-rmse{val/rmse:.4f}',
                auto_insert_metric_name=False,
                save_last=True
            )
        )
    return callbacks

def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    root = get_root()
    config = load_config(root, args.config)
    data_config = load_data_config(root, config.get('data_config'))
    use_volume = args.use_volume or config.get('use_volume', False)
    train_transform, val_transform, test_transform = build_transforms(config, use_volume)
    model, normalize = load_model(config, args.logger_type, root)
    logger = get_logger(args.logger_type, config.get('name', args.expname), args, config)
    data_module = CMambaDataModule(
        data_config,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        distributed_sampler=True,
        num_workers=args.num_workers,
        normalize=normalize,
        window_size=model.window_size,
    )
    callbacks = get_callbacks(args.save_checkpoints)
    max_epochs = config.get('max_epochs', args.max_epochs)
    model.set_normalization_coeffs(data_module.factors)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=max_epochs,
        enable_checkpointing=args.save_checkpoints,
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(model, datamodule=data_module)
    if args.save_checkpoints and callbacks:
        trainer.test(model, datamodule=data_module, ckpt_path=callbacks[0].best_model_path)

if __name__ == "__main__":
    main()
