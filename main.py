import os
import sys
import pathlib

import pytorch_lightning as pl
import warnings
import yaml
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from data_utils.data_transforms import DataTransform
from pl_modules.data_module import CMambaDataModule
from utils import io_tools

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))
warnings.simplefilter(action="ignore", category=FutureWarning)

ROOT = io_tools.get_root(__file__, num_returns=2)


def get_args():
    parser = ArgumentParser()
    # ...existing code for parser.add_argument...
    # ...existing code...
    args = parser.parse_args()
    return args


def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    path = os.path.join(log_dir, "hparams.yaml")
    if os.path.exists(path):
        return
    with open(path, "w") as f:
        yaml.dump(save_dict, f)


def load_config(args):
    config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/training/{args.config}.yaml"
    )
    data_config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml"
    )
    return config, data_config


def get_transforms(config, use_volume):
    features = config.get("additional_features", [])
    train_transform = DataTransform(
        is_train=True, use_volume=use_volume, additional_features=features
    )
    val_transform = DataTransform(
        is_train=False, use_volume=use_volume, additional_features=features
    )
    test_transform = DataTransform(
        is_train=False, use_volume=use_volume, additional_features=features
    )
    return train_transform, val_transform, test_transform


def load_model(config, logger_type):
    arch_config = io_tools.load_config_from_yaml("configs/models/archs.yaml")
    model_arch = config.get("model")
    model_config_path = f"{ROOT}/configs/models/{arch_config.get(model_arch)}"
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get("normalize", False)
    hyperparams = config.get("hyperparams")
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get("params")[key] = hyperparams.get(key)
    model_config.get("params")["logger_type"] = logger_type
    model = io_tools.instantiate_from_config(model_config)
    model.cuda()
    model.train()
    return model, normalize


def get_logger(args, config):
    name = config.get("name", args.expname)
    tmp = vars(args).copy()
    tmp.update(config)
    if args.logger_type == "tb":
        logger = TensorBoardLogger("logs", name=name)
        logger.log_hyperparams(args)
    elif args.logger_type == "wandb":
        logger = pl.loggers.WandbLogger(project=args.expname, config=tmp)
    else:
        raise ValueError("Unknown logger type.")
    return logger


def get_callbacks(args):
    callbacks = []
    if args.save_checkpoints:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/rmse",
            mode="min",
            filename="epoch{epoch}-val-rmse{val/rmse:.4f}",
            auto_insert_metric_name=False,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
    return callbacks


def get_trainer(args, logger, callbacks, max_epochs):
    return pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=max_epochs,
        enable_checkpointing=args.save_checkpoints,
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
    )


def main():
    args = get_args()
    pl.seed_everything(args.seed)
    config, data_config = load_config(args)
    use_volume = args.use_volume or config.get("use_volume")
    train_transform, val_transform, test_transform = get_transforms(
        config, use_volume
    )
    model, normalize = load_model(config, args.logger_type)
    logger = get_logger(args, config)
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
    callbacks = get_callbacks(args)
    max_epochs = config.get("max_epochs", args.max_epochs)
    model.set_normalization_coeffs(data_module.factors)
    trainer = get_trainer(args, logger, callbacks, max_epochs)
    trainer.fit(model, datamodule=data_module)
    if args.save_checkpoints and callbacks:
        checkpoint_callback = callbacks[0]
        trainer.test(
            model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path
        )


if __name__ == "__main__":
    main()