"""
Forecasting Explainer Module

This module provides explanation functionality for time series forecasts,
analyzing model performance, data characteristics, and forecast insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ForecastExplanation:
    """
    Container for forecast explanation data.
    """
    product_id: str
    model_performance: Dict[str, float]
    data_insights: Dict[str, Any]
    forecast_summary: Dict[str, Any]
    model_comparison: Dict[str, float]
    explanation_text: str


class ForecastExplainer:
    """
    Generates explanations for time series forecasts by analyzing
    historical data patterns, model performance, and forecast characteristics.
    """
    
    def __init__(self):
        self.explanation_cache = {}
    
    def generate_explanation(
        self, 
        df: pd.DataFrame, 
        forecast_df: pd.DataFrame, 
        product_id: str
    ) -> ForecastExplanation:
        """
        Generate comprehensive explanation for a forecast.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data with columns: unique_id, ds, y
        forecast_df : pd.DataFrame  
            Forecast results with predictions from multiple models
        product_id : str
            Unique identifier for the product/series
            
        Returns
        -------
        ForecastExplanation
            Detailed explanation of the forecast
        """
        # Filter data for the specific product
        product_data = df[df['unique_id'] == product_id].copy()
        product_forecast = forecast_df[forecast_df['unique_id'] == product_id].copy()
        
        if product_data.empty or product_forecast.empty:
            return self._create_empty_explanation(product_id)
        
        # Analyze historical data patterns
        data_insights = self._analyze_data_patterns(product_data)
        
        # Analyze forecast characteristics  
        forecast_summary = self._analyze_forecast_patterns(product_forecast)
        
        # Compare model performances
        model_comparison = self._compare_models(product_forecast)
        
        # Calculate model performance metrics
        model_performance = self._calculate_performance_metrics(product_data, product_forecast)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            product_id, data_insights, forecast_summary, model_comparison, model_performance
        )
        
        return ForecastExplanation(
            product_id=product_id,
            model_performance=model_performance,
            data_insights=data_insights,
            forecast_summary=forecast_summary,
            model_comparison=model_comparison,
            explanation_text=explanation_text
        )
    
    def _analyze_data_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in historical data."""
        data = data.sort_values('ds')
        y_values = data['y'].values
        
        patterns = {
            'data_points': len(data),
            'mean_value': float(np.mean(y_values)),
            'std_value': float(np.std(y_values)),
            'min_value': float(np.min(y_values)),
            'max_value': float(np.max(y_values)),
            'trend': self._detect_trend(y_values),
            'seasonality': self._detect_seasonality(y_values),
            'volatility': float(np.std(np.diff(y_values)) / np.mean(y_values)) if np.mean(y_values) != 0 else 0,
            'date_range': {
                'start': data['ds'].min().strftime('%Y-%m-%d'),
                'end': data['ds'].max().strftime('%Y-%m-%d')
            }
        }
        
        return patterns
    
    def _analyze_forecast_patterns(self, forecast_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in forecast results."""
        # Get forecast columns (exclude metadata columns)
        forecast_cols = [col for col in forecast_df.columns 
                        if col not in ['unique_id', 'ds'] and not col.endswith(('-lo-80', '-lo-90', '-hi-80', '-hi-90'))]
        
        summary = {
            'forecast_horizon': len(forecast_df),
            'models_used': forecast_cols,
            'forecast_range': {},
            'confidence_intervals': {}
        }
        
        # Analyze each model's forecasts
        for col in forecast_cols:
            if col in forecast_df.columns:
                values = forecast_df[col].values
                summary['forecast_range'][col] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values))
                }
                
                # Check for confidence intervals
                lo_80_col = f"{col}-lo-80"
                hi_80_col = f"{col}-hi-80"
                if lo_80_col in forecast_df.columns and hi_80_col in forecast_df.columns:
                    ci_width = np.mean(forecast_df[hi_80_col] - forecast_df[lo_80_col])
                    summary['confidence_intervals'][col] = float(ci_width)
        
        return summary
    
    def _compare_models(self, forecast_df: pd.DataFrame) -> Dict[str, float]:
        """Compare forecast values across different models."""
        forecast_cols = [col for col in forecast_df.columns 
                        if col not in ['unique_id', 'ds'] and not col.endswith(('-lo-80', '-lo-90', '-hi-80', '-hi-90'))]
        
        comparison = {}
        
        if len(forecast_cols) > 1:
            # Calculate variance across models for each time point
            forecast_values = forecast_df[forecast_cols].values
            variance_across_models = np.var(forecast_values, axis=1)
            
            for col in forecast_cols:
                if col in forecast_df.columns:
                    values = forecast_df[col].values
                    comparison[col] = {
                        'mean_forecast': float(np.mean(values)),
                        'forecast_trend': self._detect_trend(values),
                        'relative_variance': float(np.mean(variance_across_models))
                    }
        
        return comparison
    
    def _calculate_performance_metrics(self, historical_data: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics where possible."""
        metrics = {}
        
        # Basic statistics
        if not historical_data.empty:
            last_value = historical_data['y'].iloc[-1]
            last_date = historical_data['ds'].iloc[-1]
            
            metrics['last_historical_value'] = float(last_value)
            metrics['last_historical_date'] = last_date.strftime('%Y-%m-%d')
            
            # Calculate growth rate from historical data
            if len(historical_data) > 1:
                first_value = historical_data['y'].iloc[0]
                total_periods = len(historical_data)
                if first_value != 0:
                    growth_rate = ((last_value / first_value) ** (1/total_periods) - 1) * 100
                    metrics['historical_growth_rate'] = float(growth_rate)
        
        return metrics
    
    def _detect_trend(self, values: np.ndarray) -> str:
        """Detect trend direction in time series values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend detection
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _detect_seasonality(self, values: np.ndarray) -> str:
        """Basic seasonality detection."""
        if len(values) < 12:
            return "insufficient_data_for_seasonality"
        
        # Simple approach: check for patterns in differences
        try:
            monthly_diffs = []
            for i in range(12, len(values)):
                monthly_diffs.append(abs(values[i] - values[i-12]))
            
            if monthly_diffs:
                avg_monthly_diff = np.mean(monthly_diffs)
                overall_std = np.std(values)
                
                if avg_monthly_diff < overall_std * 0.5:
                    return "strong_seasonal"
                elif avg_monthly_diff < overall_std:
                    return "moderate_seasonal"
                else:
                    return "weak_seasonal"
        except:
            pass
        
        return "unknown"
    
    def _generate_explanation_text(
        self, 
        product_id: str, 
        data_insights: Dict[str, Any], 
        forecast_summary: Dict[str, Any],
        model_comparison: Dict[str, float],
        model_performance: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation text."""
        
        explanation_parts = []
        
        # Product overview
        explanation_parts.append(f"Forecast Explanation for {product_id}:")
        explanation_parts.append("=" * 50)
        
        # Data characteristics
        explanation_parts.append(f"\nHistorical Data Analysis:")
        explanation_parts.append(f"- Data points: {data_insights['data_points']}")
        explanation_parts.append(f"- Date range: {data_insights['date_range']['start']} to {data_insights['date_range']['end']}")
        explanation_parts.append(f"- Average value: {data_insights['mean_value']:.2f}")
        explanation_parts.append(f"- Value range: {data_insights['min_value']:.2f} to {data_insights['max_value']:.2f}")
        explanation_parts.append(f"- Trend: {data_insights['trend']}")
        explanation_parts.append(f"- Seasonality: {data_insights['seasonality']}")
        explanation_parts.append(f"- Volatility: {data_insights['volatility']:.3f}")
        
        # Forecast overview
        explanation_parts.append(f"\nForecast Summary:")
        explanation_parts.append(f"- Forecast horizon: {forecast_summary['forecast_horizon']} periods")
        explanation_parts.append(f"- Models used: {', '.join(forecast_summary['models_used'])}")
        
        # Model-specific insights
        if forecast_summary['forecast_range']:
            explanation_parts.append(f"\nModel Forecasts:")
            for model, ranges in forecast_summary['forecast_range'].items():
                explanation_parts.append(f"- {model}: {ranges['mean']:.2f} (range: {ranges['min']:.2f} - {ranges['max']:.2f})")
        
        # Performance insights
        if model_performance:
            explanation_parts.append(f"\nPerformance Insights:")
            if 'last_historical_value' in model_performance:
                explanation_parts.append(f"- Last historical value: {model_performance['last_historical_value']:.2f}")
            if 'historical_growth_rate' in model_performance:
                explanation_parts.append(f"- Historical growth rate: {model_performance['historical_growth_rate']:.2f}% per period")
        
        # Confidence intervals
        if forecast_summary['confidence_intervals']:
            explanation_parts.append(f"\nConfidence Intervals (80%):")
            for model, ci_width in forecast_summary['confidence_intervals'].items():
                explanation_parts.append(f"- {model}: ±{ci_width:.2f}")
        
        return "\n".join(explanation_parts)
    
    def _create_empty_explanation(self, product_id: str) -> ForecastExplanation:
        """Create explanation for cases with insufficient data."""
        return ForecastExplanation(
            product_id=product_id,
            model_performance={},
            data_insights={'error': 'No data available'},
            forecast_summary={'error': 'No forecast available'},
            model_comparison={},
            explanation_text=f"No explanation available for {product_id} - insufficient data."
        )