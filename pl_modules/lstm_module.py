import torch.nn as nn
from models.lstm import LSTM
from .base_module import BaseModule
    

class LSTMModule(BaseModule):

    def __init__(
        self,
        num_features=5,
        hidden_size=64,
        window_size=14,
        num_layers=1,
        bidirectional=False,
        lr=0.0002, 
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0, 
        logger_type=None,
        y_key='Close',
        optimizer='adam',
        mode='default',
        loss='rmse',
        **kwargs
    ): 
        super().__init__(lr=lr,
                         lr_step_size=lr_step_size,
                         lr_gamma=lr_gamma,
                         weight_decay=weight_decay,
                         logger_type=logger_type,
                         y_key=y_key,
                         optimizer=optimizer,
                         mode=mode,
                         window_size=window_size,
                         loss=loss,
                         )

        self.model = LSTM(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )