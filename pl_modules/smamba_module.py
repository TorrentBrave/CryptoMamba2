import torch.nn as nn
from models.smamba import SMamba
from .base_module import BaseModule
    

class SMambaModule(BaseModule):

    def __init__(
        self,
        num_features,
        seq_len,
        use_norm,
        d_model,
        d_state,
        d_ff,
        dropout,
        activation,
        e_layers,
        lr=0.0002, 
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0, 
        logger_type=None,
        window_size=14,
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

        self.model = SMamba(
            num_features=num_features,
            seq_len=seq_len,
            use_norm=use_norm,
            d_model=d_model,
            d_state=d_state,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            e_layers=e_layers,
            **kwargs
        )