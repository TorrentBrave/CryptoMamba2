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
print(ROOT)
print(__file__)
print(os.path.dirname(pathlib.Path(__file__).parent.absolute()))
print(pathlib.Path(__file__).parent.absolute())


# print(pathlib.Path('dd.py'))
# print(os.stat('/home/yuki/snap/code/CryptoMamba/scripts/dd.py'))