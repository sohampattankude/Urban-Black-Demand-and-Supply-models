"""
LSTM model architecture for demand forecasting.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LSTMDemandModel(Model):
    """BiLSTM architecture for multi-horizon demand forecasting."""
    
    def __init__(self, 
                 lstm_units_l1: int = 64,
                 lstm_units_l2: int = 32,
                 dropout_rate: float = 0.2,
                 num_horizons: int = 4,
                 seq_length: int = 24,
                 num_lag_features: int = 20):
        """
        Initialize LSTM model.
        
        Args:
            lstm_units_l1: Units in first LSTM layer
            lstm_units_l2: Units in second LSTM layer
            dropout_rate: Dropout rate
            num_horizons: Number of prediction horizons
            seq_length: Sequence length (in time steps)
            num_lag_features: Number of lag features
        """
        super().__init__()
        
        self.lstm_units_l1 = lstm_units_l1
        self.lstm_units_l2 = lstm_units_l2
        self.num_horizons = num_horizons
        
        # Input layers
        self.lag_input = layers.Input(shape=(seq_length, num_lag_features), name='lag_features')
        self.temporal_input = layers.Input(shape=(10,), name='temporal_features')
        self.supply_input = layers.Input(shape=(5,), name='supply_features')
        
        # LSTM tower
        x = layers.Bidirectional(
            layers.LSTM(lstm_units_l1, return_sequences=True, dropout=dropout_rate),
            merge_mode='concat'
        )(self.lag_input)
        
        x = layers.Bidirectional(
            layers.LSTM(lstm_units_l2, return_sequences=False, dropout=dropout_rate),
            merge_mode='concat'
        )(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Concatenate with other features
        x = layers.Concatenate()([x, self.temporal_input, self.supply_input])
        
        # Hidden layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        
        # Output heads (one per horizon)
        outputs = []
        horizon_names = ['horizon_15', 'horizon_30', 'horizon_60', 'horizon_120']
        for i, h_name in enumerate(horizon_names[:num_horizons]):
            out = layers.Dense(1, activation='relu', name=h_name)(x)
            outputs.append(out)
        
        self.model = Model(
            inputs=[self.lag_input, self.temporal_input, self.supply_input],
            outputs=outputs
        )
        logger.info(f"✅ Created LSTM model with {num_horizons} horizons")
    
    def call(self, inputs, training=False):
        """Forward pass."""
        return self.model(inputs, training=training)
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile model with weighted loss for different horizons."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        loss_dict = {}
        loss_weights_dict = {}
        metrics = ['mae']
        
        horizons = ['horizon_15', 'horizon_30', 'horizon_60', 'horizon_120']
        weights = [1.0, 0.8, 0.6, 0.4]
        
        for horizon, weight in zip(horizons, weights):
            loss_dict[horizon] = keras.losses.MeanSquaredError()
            loss_weights_dict[horizon] = weight
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics
        )
        
        logger.info("✅ Compiled LSTM model")
    
    def get_summary(self):
        """Print model architecture summary."""
        return self.model.summary()
    
    def save_model(self, path: str):
        """Save model to disk."""
        self.model.save(path)
        logger.info(f"✅ Saved LSTM model to {path}")
    
    @staticmethod
    def load_model(path: str):
        """Load model from disk."""
        model = keras.models.load_model(path)
        logger.info(f"✅ Loaded LSTM model from {path}")
        return model
