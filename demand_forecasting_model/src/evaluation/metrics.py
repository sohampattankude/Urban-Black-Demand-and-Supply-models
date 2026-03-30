"""
Evaluation metrics for demand forecasting models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute MAPE (Mean Absolute Percentage Error).
    Ignores division by zero with epsilon.
    """
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

def quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Compute quantile loss.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Quantile level (0.1, 0.5, 0.9, etc.)
    
    Returns:
        Quantile loss value
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))

def evaluate_predictions(y_true, y_pred, horizon_name=''):
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        horizon_name: Name of the horizon (for logging)
    
    Returns:
        Dictionary with computed metrics
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Quantile losses
    q10_loss = quantile_loss(y_true, y_pred, quantile=0.1)
    q50_loss = quantile_loss(y_true, y_pred, quantile=0.5)
    q90_loss = quantile_loss(y_true, y_pred, quantile=0.9)
    
    # Error percentiles
    abs_errors = np.abs(y_true - y_pred)
    p50_error = np.percentile(abs_errors, 50)
    p90_error = np.percentile(abs_errors, 90)
    p95_error = np.percentile(abs_errors, 95)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'q10_loss': q10_loss,
        'q50_loss': q50_loss,
        'q90_loss': q90_loss,
        'p50_error': p50_error,
        'p90_error': p90_error,
        'p95_error': p95_error,
        'count': len(y_true),
        'mean_prediction': float(np.mean(y_pred)),
        'mean_actual': float(np.mean(y_true))
    }
    
    # Print summary
    if horizon_name:
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics — {horizon_name}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics")
        print(f"{'='*60}")
    
    print(f"MAE:          {mae:.2f} requests")
    print(f"RMSE:         {rmse:.2f} requests")
    print(f"MAPE:         {mape:.2f}%")
    print(f"p50 Error:    {p50_error:.2f} requests")
    print(f"p90 Error:    {p90_error:.2f} requests")
    print(f"Q10 Loss:     {q10_loss:.4f}")
    print(f"Q50 Loss:     {q50_loss:.4f}")
    print(f"Q90 Loss:     {q90_loss:.4f}")
    print(f"Samples:      {len(y_true):,}")
    print(f"{'='*60}")
    
    return metrics

class ValidationMetrics:
    """Compute validation metrics across different dimensions."""
    
    @staticmethod
    def metrics_by_time_of_day(y_true, y_pred, hours):
        """Compute metrics grouped by hour of day."""
        metrics_by_hour = {}
        unique_hours = np.sort(np.unique(hours))
        
        for hour in unique_hours:
            mask = hours == hour
            if mask.sum() > 0:
                metrics_by_hour[int(hour)] = evaluate_predictions(
                    y_true[mask], 
                    y_pred[mask],
                    horizon_name=f"Hour {hour}"
                )
        
        return metrics_by_hour
    
    @staticmethod
    def metrics_by_demand_level(y_true, y_pred, bins=5):
        """Compute metrics grouped by demand level."""
        # Bin by actual demand
        demand_bins = np.quantile(y_true, np.linspace(0, 1, bins + 1))
        binned = np.digitize(y_true, demand_bins)
        
        metrics_by_bin = {}
        for bin_idx in range(1, bins + 1):
            mask = binned == bin_idx
            if mask.sum() > 0:
                low = demand_bins[bin_idx - 1]
                high = demand_bins[bin_idx]
                metrics_by_bin[f"Demand {low:.1f}-{high:.1f}"] = evaluate_predictions(
                    y_true[mask],
                    y_pred[mask],
                    horizon_name=f"Demand {low:.1f}-{high:.1f}"
                )
        
        return metrics_by_bin
