import os
import sys
import pathlib
from argparse import ArgumentParser
from datetime import datetime
import warnings

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import io_tools
from utils.trade import trade
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform

warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_theme(style='whitegrid', context='paper', font_scale=2)
palette = sns.color_palette('muted')

LABEL_DICT = {
    'cmamba': 'CryptoMamba',
    'lstm': 'LSTM',
    'lstm_bi': 'Bi-LSTM',
    'gru': 'GRU',
    'smamba': 'S-Mamba',
    'itransformer': 'iTransformer',
}


def get_root():
    """获取项目根目录"""
    return io_tools.get_root(__file__, num_returns=2)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default='gpu', help="The type of accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of computing devices.")
    parser.add_argument("--seed", type=int, default=23, help="Logging directory.")
    parser.add_argument("--expname", type=str, default='Cmamba', help="Experiment name. Reconstructions will be saved under this folder.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--logger_type", default='tb', type=str, help="Path to config file.")
    parser.add_argument("--ckpt_path", default=None, type=str, help="Path to config file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--balance", type=float, default=100, help="initial money")
    parser.add_argument("--risk", type=float, default=2)
    parser.add_argument("--split", type=str, default='test', choices={'test', 'val', 'train'})
    parser.add_argument("--trade_mode", type=str, default='smart', choices={'smart', 'smart_w_short', 'vanilla', 'no_strategy'})
    return parser.parse_args()


def load_model(config, ckpt_path, config_name=None, root=None):
    if root is None:
        root = get_root()
    if ckpt_path is None:
        ckpt_path = f'{root}/checkpoints/{config_name}.ckpt'
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{root}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get('normalize', False)
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'))
    model.cuda()
    model.eval()
    return model, normalize


def init_dirs(result_dir):
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)


def max_drawdown(prices):
    prices = np.array(prices)
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    mdd = drawdown.min()
    return -mdd


def run_model(model, dataloader, factors=None):
    target_list, preds_list, timetamps = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get('Timestamp').numpy().reshape(-1)
            target = batch.get(model.y_key).numpy().reshape(-1)
            features = batch.get('features').to(model.device)
            preds = model(features).cpu().numpy().reshape(-1)
            target_list += [float(x) for x in list(target)]
            preds_list += [float(x) for x in list(preds)]
            if factors is not None:
                timetamps += [float(x) for x in list(batch.get('Timestamp_orig').numpy().reshape(-1))]
            else:
                timetamps += [float(x) for x in list(ts)]
    if factors is not None:
        scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
        shift = factors.get(model.y_key).get('min')
        target_list = [x * scale + shift for x in target_list]
        preds_list = [x * scale + shift for x in preds_list]
    return np.asarray(timetamps), np.asarray(target_list), np.asarray(preds_list)


def get_data_module(config, data_config, batch_size, num_workers, normalize):
    use_volume = config.get('use_volume', False)
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    return CMambaDataModule(
        data_config,
        train_transform=test_transform,
        val_transform=test_transform,
        test_transform=test_transform,
        batch_size=batch_size,
        distributed_sampler=False,
        num_workers=num_workers,
        normalize=normalize,
    )


def simulate_trade_and_plot(
    config_list, args, colors, root, label_dict=LABEL_DICT
):
    plt.figure(figsize=(15, 10))
    balances, all_tmp = [], []
    for conf, c in zip(config_list, colors):
        config = io_tools.load_config_from_yaml(f'{root}/configs/training/{conf}.yaml')
        name = config.get('name', args.expname)
        result_dir = f'{root}/Results/{name}/{args.config}' if len(config_list) == 1 else f'{root}/Results/all/'
        init_dirs(result_dir)
        data_config = io_tools.load_config_from_yaml(f"{root}/configs/data_configs/{config.get('data_config')}.yaml")
        model, normalize = load_model(config, args.ckpt_path, config_name=conf, root=root)
        data_module = get_data_module(config, data_config, args.batch_size, args.num_workers, normalize)
        if args.split == 'test':
            loader = data_module.test_dataloader()
        elif args.split == 'val':
            loader = data_module.val_dataloader()
        else:
            loader = data_module.train_dataloader()
        factors = data_module.factors if normalize else None
        timstamps, targets, preds = run_model(model, loader, factors)
        data = loader.dataset.data
        time_key = 'Timestamp_orig' if normalize else 'Timestamp'
        if normalize:
            scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
            shift = factors.get(model.y_key).get('min')
            data[model.y_key] = data[model.y_key] * scale + shift
        balance, balance_in_time = trade(
            data, time_key, timstamps, targets, preds,
            balance=args.balance, mode=args.trade_mode, risk=args.risk, y_key=model.y_key
        )
        balances.append((conf, balance, balance_in_time))
        print(f'{conf} -- Final balance: {round(balance, 2)}')
        print(f'{conf} -- Maximum Draw Down : {round(max_drawdown(balance_in_time) * 100, 2)}')
        label = label_dict.get(conf.replace("_nv", "").replace("_v", ""))
        tmp = [timstamps[0] - 24 * 60 * 60] + list(timstamps)
        tmp = [datetime.fromtimestamp(int(x)) for x in tmp]
        all_tmp.append(tmp)
        sns.lineplot(x=tmp, y=balance_in_time, color=c, zorder=0, linewidth=2.5, label=label)
    return balances, all_tmp


def plot_and_save(balances, all_tmp, config_list, args, root, config):
    name = config.get('name', args.expname)
    if args.trade_mode == 'no_strategy':
        plot_path = f'./balance_{args.split}.jpg'
    else:
        if len(config_list) == 1:
            plot_path = f'{root}/Results/{name}/{args.config}/balance_{args.split}_{args.trade_mode}.jpg'
        else:
            plot_path = f'{root}/Results/all/balance_{args.config}_{args.split}_{args.trade_mode}.jpg'
    plt.xticks(rotation=30)
    plt.axhline(y=100, color='r', linestyle='--')
    if len(config_list) == 1:
        ax = plt.gca()
        ax.get_legend().remove()
        plt.title(f'Balance in time (final: {round(balances[0][1], 2)})')
    else:
        plt.title(f'Net Worth in Time')
    plt.xlim([all_tmp[0][0], all_tmp[0][-1]])
    plt.ylabel('Balance ($)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')


def main():
    args = get_args()
    root = get_root()
    colors = ['darkblue', 'yellowgreen', 'crimson', 'darkviolet', 'orange', 'magenta']
    if args.config == 'all':
        config_list = [x.replace('.ckpt', '') for x in os.listdir(f'{root}/checkpoints/') if '_nv.ckpt' in x]
    elif args.config == 'all_v':
        config_list = [x.replace('.ckpt', '') for x in os.listdir(f'{root}/checkpoints/') if '_v.ckpt' in x]
    else:
        config_list = [args.config]
        colors = ['darkblue']
    balances, all_tmp = simulate_trade_and_plot(config_list, args, colors, root)
    # 取第一个config用于plot_and_save
    config = io_tools.load_config_from_yaml(f'{root}/configs/training/{config_list[0]}.yaml')
    plot_and_save(balances, all_tmp, config_list, args, root, config)


if __name__ == '__main__':
    main()