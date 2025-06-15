import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import time
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from utils import io_tools
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
from utils.trade import buy_sell_vanilla, buy_sell_smart
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


ROOT = io_tools.get_root(__file__, num_returns=2)

class ConfigLoader:
    @staticmethod
    def load_training_config(config_name):
        return io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{config_name}.yaml')

    @staticmethod
    def load_data_config(data_config_name):
        return io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{data_config_name}.yaml")

    @staticmethod
    def load_model_config(model_arch):
        arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
        model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
        return io_tools.load_config_from_yaml(model_config_path)

class ModelHandler:
    @staticmethod
    def load_model(config, ckpt_path):
        model_config = ConfigLoader.load_model_config(config.get('model'))
        normalize = model_config.get('normalize', False)
        model_class = io_tools.get_obj_from_str(model_config.get('target'))
        model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'))
        model.cuda()
        return model, normalize

class DataPreparer:
    @staticmethod
    def prepare_data(data_path, date, use_volume, normalize, data_module, model_y_key):
        data = pd.read_csv(data_path)
        if 'Date' in data.keys():
            data['Timestamp'] = [float(time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple())) for x in data['Date']]
        data = data.sort_values(by='Timestamp').reset_index(drop=True)

        if date is None:
            end_ts = max(data['Timestamp']) + 24 * 60 * 60
        else:
            end_ts = int(time.mktime(datetime.strptime(date, "%Y-%m-%d").timetuple()))
        start_ts = end_ts - 14 * 24 * 60 * 60 - 60 * 60
        pred_date = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")
        data = data[(data['Timestamp'] < end_ts) & (data['Timestamp'] >= start_ts - 60 * 60)]

        features, scale_pred, shift_pred = DataPreparer._extract_features(
            data, use_volume, normalize, data_module, model_y_key
        )
        return features, scale_pred, shift_pred, pred_date

    @staticmethod
    def _extract_features(data, use_volume, normalize, data_module, model_y_key):
        features = {}
        key_list = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        if use_volume:
            key_list.append('Volume')
        scale_pred, shift_pred = 1, 0
        for key in key_list:
            tmp = list(data.get(key))
            if normalize:
                scale = data_module.factors.get(key).get('max') - data_module.factors.get(key).get('min')
                shift = data_module.factors.get(key).get('min')
            else:
                scale = 1
                shift = 0
            if key == 'Volume':
                tmp = [x / 1e9 for x in tmp]
            tmp = [(x - shift) / scale for x in tmp]
            features[key] = torch.tensor(tmp).reshape(1, -1)
            if key == model_y_key:
                scale_pred = scale
                shift_pred = shift
        return features, scale_pred, shift_pred

class TradeReporter:
    def __init__(self, txt_file):
        self.txt_file = txt_file

    def print_and_write(self, txt, add_new_line=True):
        print(txt)
        if add_new_line:
            self.txt_file.write(f'{txt}\n')
        else:
            self.txt_file.write(txt)

    def report_prediction(self, pred_date, pred, today):
        self.print_and_write(f'Prediction date: {pred_date}\nPrediction: {round(pred, 2)}\nToday value: {round(today, 2)}')

    def report_smart_trade(self, today, pred, risk):
        b, s = buy_sell_smart(today, pred, 100, 100, risk=risk)
        if b < 100:
            tmp = round((100 - b), 2)
            self.print_and_write(f'Smart trade: {tmp}% buy')
        if s < 100:
            tmp = round((100 - s), 2)
            self.print_and_write(f'Smart trade: {tmp}% sell')

    def report_vanilla_trade(self, today, pred):
        b, s = buy_sell_vanilla(today, pred, 100, 100)
        if b < 100:
            assert b == 0
            self.print_and_write(f'Vanilla trade: buy')
        elif s < 100:
            assert s == 0
            self.print_and_write(f'Vanilla trade: sell')
        else:
            self.print_and_write(f'Vanilla trade: -')

# 在使用终端脚本运行命令时，获取命令行参数，根据不同的参数设置加载不同的配置文件和模型，核心主文件和模型创建于训练文件中的必备函数
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default='gpu', help="The type of accelerator.")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1, help="Number of computing devices.")
    parser.add_argument("--seed", type=int, default=23, help="Logging directory.")
    parser.add_argument("--config", type=str, default='cmamba_v', help="Path to config file.")
    parser.add_argument('--use_volume', default=False, action='store_true')
    parser.add_argument("--data_path", default='data/one_day_pred.csv', type=str, help="Path to config file.")
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--risk", default=2, type=int)
    return parser.parse_args()

def init_dirs(args, date):
    path = f'{ROOT}/Predictions/{args.config}/'
    os.makedirs(path, exist_ok=True)
    txt_file = open(f'{path}/{date}.txt', 'w')
    return txt_file

def main():
    args = get_args()
    config = ConfigLoader.load_training_config(args.config)
    data_config = ConfigLoader.load_data_config(config.get('data_config'))
    use_volume = config.get('use_volume', args.use_volume)
    model, normalize = ModelHandler.load_model(config, args.ckpt_path)

    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    data_module = CMambaDataModule(
        data_config,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=1,
        distributed_sampler=False,
        num_workers=1,
        normalize=normalize,
    )

    features, scale_pred, shift_pred, pred_date = DataPreparer.prepare_data(
        args.data_path, args.date, use_volume, normalize, data_module, model.y_key
    )
    x = torch.cat([features.get(k) for k in features.keys()], dim=0)
    close_idx = -2 if use_volume else -1
    today = float(x[close_idx, -1])

    with torch.no_grad():
        pred = float(model(x[None, ...].cuda()).cpu()) * scale_pred + shift_pred

    txt_file = init_dirs(args, pred_date)
    reporter = TradeReporter(txt_file)
    print('')
    reporter.report_prediction(pred_date, pred, today)
    reporter.report_smart_trade(today, pred, args.risk)
    reporter.report_vanilla_trade(today, pred)
    txt_file.close()

if __name__ == "__main__":
    main()