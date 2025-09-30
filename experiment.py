
import numpy as np
import sys
import logging
from typing import Tuple, List, Optional
from scipy.stats import norm
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesDataGenerator:
    """Generate synthetic time series data with distribution shifts"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_stationary_series(self, n_samples: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """Generate stationary time series data"""
        return np.random.normal(mean, std, n_samples)
    
    def generate_shifting_series(self, n_samples: int, shift_times: List[int], 
                               shift_magnitudes: List[float]) -> np.ndarray:
        """Generate time series with distribution shifts at specified times"""
        if len(shift_times) != len(shift_magnitudes):
            raise ValueError("shift_times and shift_magnitudes must have same length")
        
        series = np.zeros(n_samples)
        current_mean = 0.0
        current_std = 1.0
        
        shift_times_sorted = sorted(shift_times)
        shift_magnitudes_sorted = [shift_magnitudes[shift_times.index(t)] for t in shift_times_sorted]
        
        start_idx = 0
        for i, shift_time in enumerate(shift_times_sorted):
            end_idx = min(shift_time, n_samples)
            series[start_idx:end_idx] = np.random.normal(current_mean, current_std, end_idx - start_idx)
            
            if i < len(shift_magnitudes_sorted):
                current_mean += shift_magnitudes_sorted[i]
                current_std *= 1.1  # Slight increase in variance
            
            start_idx = end_idx
        
        # Fill remaining series
        if start_idx < n_samples:
            series[start_idx:] = np.random.normal(current_mean, current_std, n_samples - start_idx)
        
        return series

class LPConformalPredictor:
    """Lévy-Prokhorov Robust Conformal Prediction for Time Series"""
    
    def __init__(self, epsilon: float, rho: float, alpha: float = 0.1):
        self.epsilon = epsilon  # Local perturbation parameter
        self.rho = rho          # Global perturbation parameter  
        self.alpha = alpha      # Miscoverage level
        self.calibration_scores = None
        self.quantile_wc = None
    
    def compute_scores(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores (absolute errors)"""
        return np.abs(predictions - targets)
    
    def fit_calibration(self, calibration_scores: np.ndarray):
        """Fit the conformal predictor using calibration scores"""
        if len(calibration_scores) == 0:
            raise ValueError("Calibration scores cannot be empty")
        
        self.calibration_scores = calibration_scores
        n = len(calibration_scores)
        
        # Compute worst-case quantile using LP theory
        beta_adjusted = self.alpha + (self.alpha - self.rho - 2) / n
        quantile_level = 1 - beta_adjusted + self.rho
        
        if quantile_level <= 0 or quantile_level >= 1:
            logger.warning(f"Adjusted quantile level {quantile_level:.4f} is outside (0,1)")
            quantile_level = np.clip(quantile_level, 0.01, 0.99)
        
        # Empirical quantile computation
        sorted_scores = np.sort(calibration_scores)
        idx = min(int(np.ceil(quantile_level * n)), n - 1)
        empirical_quantile = sorted_scores[idx]
        
        # Apply LP worst-case adjustment
        self.quantile_wc = empirical_quantile + self.epsilon
        
        logger.info(f"Fitted LP conformal predictor: epsilon={self.epsilon}, rho={self.rho}")
        logger.info(f"Empirical quantile: {empirical_quantile:.4f}, Worst-case quantile: {self.quantile_wc:.4f}")
    
    def predict_interval(self, prediction: float) -> Tuple[float, float]:
        """Generate prediction interval for a single prediction"""
        if self.quantile_wc is None:
            raise ValueError("Predictor must be fitted before making predictions")
        
        lower_bound = prediction - self.quantile_wc
        upper_bound = prediction + self.quantile_wc
        
        return lower_bound, upper_bound
    
    def compute_coverage(self, predictions: np.ndarray, targets: np.ndarray, 
                        intervals: List[Tuple[float, float]]) -> float:
        """Compute empirical coverage of prediction intervals"""
        covered = 0
        for i, (lower, upper) in enumerate(intervals):
            if lower <= targets[i] <= upper:
                covered += 1
        return covered / len(targets)

class SimpleTimeSeriesPredictor:
    """Simple time series predictor for demonstration"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def predict(self, series: np.ndarray) -> np.ndarray:
        """Simple moving average prediction"""
        predictions = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < self.window_size:
                # For initial points, use available history
                predictions[i] = np.mean(series[:max(1, i)])
            else:
                # Moving average prediction
                predictions[i] = np.mean(series[i-self.window_size:i])
        
        return predictions

def run_experiment():
    """Main experiment function"""
    try:
        logger.info("Starting LP Conformal Prediction Time Series Experiment")
        
        # Generate synthetic time series data
        logger.info("Generating synthetic time series data...")
        generator = TimeSeriesDataGenerator(seed=42)
        
        # Create training data (stationary)
        n_train = 500
        train_series = generator.generate_stationary_series(n_train)
        
        # Create test data with distribution shifts
        n_test = 300
        shift_times = [100, 200]
        shift_magnitudes = [2.0, -1.5]  # Mean shifts
        test_series = generator.generate_shifting_series(n_test, shift_times, shift_magnitudes)
        
        logger.info(f"Generated {n_train} training samples and {n_test} test samples")
        logger.info(f"Test series has distribution shifts at times: {shift_times}")
        
        # Split data for conformal prediction
        n_calibration = 200
        calibration_series = train_series[:n_calibration]
        remaining_train = train_series[n_calibration:]
        
        # Train simple predictor
        logger.info("Training time series predictor...")
        predictor = SimpleTimeSeriesPredictor(window_size=5)
        train_predictions = predictor.predict(remaining_train)
        train_targets = remaining_train
        
        # Compute calibration scores
        calibration_predictor = SimpleTimeSeriesPredictor(window_size=5)
        calibration_predictions = calibration_predictor.predict(calibration_series)
        calibration_targets = calibration_series
        
        lp_predictor = LPConformalPredictor(epsilon=0.5, rho=0.05, alpha=0.1)
        calibration_scores = lp_predictor.compute_scores(calibration_predictions, calibration_targets)
        
        # Fit LP conformal predictor
        logger.info("Fitting LP conformal predictor...")
        lp_predictor.fit_calibration(calibration_scores)
        
        # Test on shifting time series
        logger.info("Testing on time series with distribution shifts...")
        test_predictions = predictor.predict(test_series)
        test_targets = test_series
        
        # Generate prediction intervals
        test_intervals = []
        for pred in test_predictions:
            interval = lp_predictor.predict_interval(pred)
            test_intervals.append(interval)
        
        # Compute coverage and statistics
        coverage = lp_predictor.compute_coverage(test_predictions, test_targets, test_intervals)
        
        # Compute interval widths
        interval_widths = [upper - lower for lower, upper in test_intervals]
        avg_width = np.mean(interval_widths)
        std_width = np.std(interval_widths)
        
        # Compare with standard conformal prediction
        standard_quantile = np.quantile(calibration_scores, 0.9)
        standard_intervals = [(pred - standard_quantile, pred + standard_quantile) 
                            for pred in test_predictions]
        standard_coverage = lp_predictor.compute_coverage(test_predictions, test_targets, standard_intervals)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("EXPERIMENT RESULTS")
        logger.info("="*50)
        logger.info(f"LP Conformal Prediction:")
        logger.info(f"  - Empirical Coverage: {coverage:.4f} (Target: {1-lp_predictor.alpha:.2f})")
        logger.info(f"  - Average Interval Width: {avg_width:.4f}")
        logger.info(f"  - Std of Interval Widths: {std_width:.4f}")
        logger.info(f"  - Worst-case Quantile: {lp_predictor.quantile_wc:.4f}")
        
        logger.info(f"\nStandard Conformal Prediction:")
        logger.info(f"  - Empirical Coverage: {standard_coverage:.4f}")
        logger.info(f"  - Quantile: {standard_quantile:.4f}")
        
        logger.info(f"\nLP Parameters:")
        logger.info(f"  - Epsilon (local): {lp_predictor.epsilon}")
        logger.info(f"  - Rho (global): {lp_predictor.rho}")
        logger.info(f"  - Alpha: {lp_predictor.alpha}")
        
        # Test robustness by varying LP parameters
        logger.info("\n" + "="*50)
        logger.info("ROBUSTNESS ANALYSIS")
        logger.info("="*50)
        
        epsilon_values = [0.1, 0.5, 1.0]
        rho_values = [0.01, 0.05, 0.1]
        
        robustness_results = []
        for eps in epsilon_values:
            for rho_val in rho_values:
                try:
                    robust_predictor = LPConformalPredictor(epsilon=eps, rho=rho_val, alpha=0.1)
                    robust_predictor.fit_calibration(calibration_scores)
                    
                    robust_intervals = []
                    for pred in test_predictions:
                        interval = robust_predictor.predict_interval(pred)
                        robust_intervals.append(interval)
                    
                    robust_coverage = robust_predictor.compute_coverage(
                        test_predictions, test_targets, robust_intervals)
                    
                    robust_widths = [upper - lower for lower, upper in robust_intervals]
                    avg_robust_width = np.mean(robust_widths)
                    
                    robustness_results.append({
                        'epsilon': eps,
                        'rho': rho_val,
                        'coverage': robust_coverage,
                        'avg_width': avg_robust_width,
                        'quantile': robust_predictor.quantile_wc
                    })
                    
                    logger.info(f"ε={eps:.1f}, ρ={rho_val:.2f}: "
                              f"Coverage={robust_coverage:.4f}, "
                              f"Width={avg_robust_width:.4f}")
                              
                except Exception as e:
                    logger.warning(f"Failed for ε={eps}, ρ={rho_val}: {str(e)}")
                    continue
        
        # Find best parameter combination
        if robustness_results:
            best_result = min(robustness_results, 
                            key=lambda x: abs(x['coverage'] - (1-lp_predictor.alpha)) + 0.1*x['avg_width'])
            
            logger.info(f"\nBest LP parameters:")
            logger.info(f"  - Epsilon: {best_result['epsilon']:.2f}")
            logger.info(f"  - Rho: {best_result['rho']:.2f}")
            logger.info(f"  - Coverage: {best_result['coverage']:.4f}")
            logger.info(f"  - Average Width: {best_result['avg_width']:.4f}")
        
        logger.info("\nExperiment completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        logger.error("Terminating program...")
        return 1

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run experiment
    exit_code = run_experiment()
    sys.exit(exit_code)
