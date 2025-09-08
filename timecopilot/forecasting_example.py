"""
Complete TimeCopilot Forecasting and Explanation Example

This module provides a complete working example of how to:
1. Load real time series data
2. Generate forecasts using multiple models
3. Create explanations for the forecasts
4. Handle multiple products/time series
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import explainer directly
from .explainer import ForecastExplainer


class TimeCopilotForecastingPipeline:
    """
    Complete pipeline for forecasting and explanation generation.
    """
    
    def __init__(self, models: Optional[List] = None):
        """
        Initialize the forecasting pipeline.
        
        Parameters
        ----------
        models : List, optional
            List of forecasting models to use. If None, uses default models.
        """
        # Lazy import to avoid dependency issues
        try:
            from timecopilot import TimeCopilotForecaster
            from timecopilot.models.stats import AutoARIMA, AutoETS, SeasonalNaive, Theta
            from timecopilot.models.prophet import Prophet
            
            if models is None:
                # Default models that are reliable and fast
                models = [
                    AutoARIMA(),
                    AutoETS(), 
                    SeasonalNaive(),
                    Theta(),
                    Prophet()
                ]
            
            self.forecaster = TimeCopilotForecaster(models=models)
            self._forecaster_available = True
            
        except ImportError as e:
            print(f"Warning: TimeCopilotForecaster not available due to missing dependencies: {e}")
            self.forecaster = None
            self._forecaster_available = False
        
        self.explainer = ForecastExplainer()
        self.forecasts = {}  # Store forecasts by product_id
        self.explanations = {}  # Store explanations by product_id
    
    def load_sample_data(self) -> pd.DataFrame:
        """
        Load sample real data for demonstration.
        
        Returns
        -------
        pd.DataFrame
            Sample time series data with multiple products
        """
        try:
            # Try to load the official TimeCopilot sample data
            df = pd.read_csv(
                "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
                parse_dates=["ds"]
            )
            
            # Create multiple products by modifying the data
            products_data = []
            
            # Original AirPassengers
            original_data = df.copy()
            original_data['unique_id'] = 'AirPassengers'
            products_data.append(original_data)
            
            # Create Product_A (modified version)
            product_a = df.copy()
            product_a['unique_id'] = 'Product_A'
            product_a['y'] = product_a['y'] * 1.5 + np.random.normal(0, 10, len(product_a))
            products_data.append(product_a)
            
            # Create Product_B (another modified version)
            product_b = df.copy()
            product_b['unique_id'] = 'Product_B'
            product_b['y'] = product_b['y'] * 0.8 + 50 + np.random.normal(0, 15, len(product_b))
            products_data.append(product_b)
            
            # Combine all products
            combined_df = pd.concat(products_data, ignore_index=True)
            
            return combined_df
            
        except Exception as e:
            print(f"Could not load remote data: {e}")
            # Fallback to generating synthetic data
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic time series data as fallback.
        
        Returns
        -------
        pd.DataFrame
            Synthetic time series data
        """
        print("Generating synthetic data as fallback...")
        
        # Generate 3 years of monthly data
        dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='M')
        
        products_data = []
        
        for product_id in ['Product_A', 'Product_B', 'Product_C']:
            # Generate realistic time series with trend and seasonality
            trend = np.linspace(100, 200, len(dates))
            seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
            noise = np.random.normal(0, 10, len(dates))
            
            # Add product-specific characteristics
            if product_id == 'Product_A':
                multiplier = 1.2
            elif product_id == 'Product_B':
                multiplier = 0.8
            else:
                multiplier = 1.0
            
            y_values = (trend + seasonality + noise) * multiplier
            
            product_data = pd.DataFrame({
                'unique_id': product_id,
                'ds': dates,
                'y': y_values
            })
            
            products_data.append(product_data)
        
        return pd.concat(products_data, ignore_index=True)
    
    def generate_forecasts(self, df: pd.DataFrame, h: int = 12, level: List[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for all products in the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data with columns: unique_id, ds, y
        h : int, default 12
            Forecast horizon (number of periods to forecast)
        level : List[int], optional
            Confidence levels for prediction intervals
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping product_id to forecast DataFrame
        """
        if not self._forecaster_available:
            print("Error: TimeCopilotForecaster not available. Cannot generate forecasts.")
            return {}
        
        if level is None:
            level = [80, 90]
        
        print("Generating forecasts...")
        
        # Generate forecasts for all data at once
        try:
            forecast_df = self.forecaster.forecast(df=df, h=h, level=level)
            
            # Split forecasts by product_id
            forecasts = {}
            for product_id in df['unique_id'].unique():
                product_forecast = forecast_df[forecast_df['unique_id'] == product_id].copy()
                if not product_forecast.empty:
                    forecasts[product_id] = product_forecast
                    print(f"Generated forecast for {product_id}: {len(product_forecast)} periods")
            
            self.forecasts = forecasts
            return forecasts
            
        except Exception as e:
            print(f"Error generating forecasts: {e}")
            return {}
    
    def generate_explanations(self, df: pd.DataFrame, product_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate explanations for the forecasts.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data
        product_ids : List[str], optional
            List of product IDs to explain. If None, explains all available forecasts.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping product_id to explanation
        """
        if not self.forecasts:
            print("No forecasts available. Please generate forecasts first.")
            return {}
        
        if product_ids is None:
            product_ids = list(self.forecasts.keys())
        
        print("Generating explanations...")
        
        explanations = {}
        for product_id in product_ids:
            if product_id in self.forecasts:
                try:
                    explanation = self.explainer.generate_explanation(
                        df=df,
                        forecast_df=self.forecasts[product_id],
                        product_id=product_id
                    )
                    explanations[product_id] = explanation
                    print(f"Generated explanation for {product_id}")
                except Exception as e:
                    print(f"Error generating explanation for {product_id}: {e}")
        
        self.explanations = explanations
        return explanations
    
    def print_summary(self, max_products: int = 3):
        """
        Print a summary of forecasts and explanations.
        
        Parameters
        ----------
        max_products : int, default 3
            Maximum number of products to display in detail
        """
        print("\n" + "="*60)
        print("TIMECOPILOT FORECASTING SUMMARY")
        print("="*60)
        
        if not self.forecasts:
            print("No forecasts available.")
            return
        
        print(f"\nTotal products forecasted: {len(self.forecasts)}")
        print(f"Explanations generated: {len(self.explanations)}")
        
        # Display detailed information for first few products
        product_ids = list(self.forecasts.keys())[:max_products]
        
        for product_id in product_ids:
            print(f"\n{'-'*50}")
            print(f"PRODUCT: {product_id}")
            print(f"{'-'*50}")
            
            # Forecast summary
            if product_id in self.forecasts:
                forecast_df = self.forecasts[product_id]
                print(f"Forecast periods: {len(forecast_df)}")
                
                # Show model columns
                model_cols = [col for col in forecast_df.columns 
                             if col not in ['unique_id', 'ds'] and not col.endswith(('-lo-80', '-lo-90', '-hi-80', '-hi-90'))]
                print(f"Models used: {', '.join(model_cols)}")
                
                # Show sample forecast values
                if not forecast_df.empty:
                    print(f"\nFirst few forecast values:")
                    display_df = forecast_df[['ds'] + model_cols].head(3)
                    for _, row in display_df.iterrows():
                        date_str = row['ds'].strftime('%Y-%m-%d')
                        print(f"  {date_str}: ", end="")
                        for col in model_cols:
                            print(f"{col}={row[col]:.2f} ", end="")
                        print()
            
            # Explanation summary
            if product_id in self.explanations:
                explanation = self.explanations[product_id]
                print(f"\nData insights:")
                data_insights = explanation.data_insights
                if 'data_points' in data_insights:
                    print(f"  - Historical data points: {data_insights['data_points']}")
                if 'trend' in data_insights:
                    print(f"  - Trend: {data_insights['trend']}")
                if 'seasonality' in data_insights:
                    print(f"  - Seasonality: {data_insights['seasonality']}")
                if 'mean_value' in data_insights:
                    print(f"  - Average value: {data_insights['mean_value']:.2f}")
        
        if len(self.forecasts) > max_products:
            remaining = len(self.forecasts) - max_products
            print(f"\n... and {remaining} more product(s)")
    
    def get_forecast_dataframe(self, product_id: str) -> Optional[pd.DataFrame]:
        """
        Get forecast DataFrame for a specific product.
        
        Parameters
        ----------
        product_id : str
            Product identifier
            
        Returns
        -------
        pd.DataFrame or None
            Forecast DataFrame for the product
        """
        return self.forecasts.get(product_id)
    
    def get_explanation(self, product_id: str) -> Optional[Any]:
        """
        Get explanation for a specific product.
        
        Parameters
        ----------
        product_id : str
            Product identifier
            
        Returns
        -------
        ForecastExplanation or None
            Explanation for the product
        """
        return self.explanations.get(product_id)
    
    def print_detailed_explanation(self, product_id: str):
        """
        Print detailed explanation for a specific product.
        
        Parameters
        ----------
        product_id : str
            Product identifier
        """
        if product_id not in self.explanations:
            print(f"No explanation available for {product_id}")
            return
        
        explanation = self.explanations[product_id]
        print(explanation.explanation_text)


def main():
    """
    Main function demonstrating the complete workflow.
    """
    print("TimeCopilot Forecasting and Explanation Demo")
    print("=" * 50)
    
    # Initialize the pipeline
    pipeline = TimeCopilotForecastingPipeline()
    
    # Load data
    print("Loading data...")
    df = pipeline.load_sample_data()
    print(f"Loaded data for {df['unique_id'].nunique()} products")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Total data points: {len(df)}")
    
    # Generate forecasts
    forecasts = pipeline.generate_forecasts(df, h=12, level=[80, 90])
    
    if not forecasts:
        print("Failed to generate forecasts")
        return
    
    # Generate explanations (limit to first 3 products for demo)
    product_ids = list(df['unique_id'].unique())[:3]
    explanations = pipeline.generate_explanations(df, product_ids)
    
    # Print summary
    pipeline.print_summary(max_products=3)
    
    # Print detailed explanation for first product
    if product_ids:
        first_product = product_ids[0]
        print(f"\n{'='*60}")
        print(f"DETAILED EXPLANATION FOR {first_product}")
        print(f"{'='*60}")
        pipeline.print_detailed_explanation(first_product)
    
    print("\nDemo completed successfully!")
    return pipeline


if __name__ == "__main__":
    # Run the main demo
    pipeline = main()
    
    # The forecasts and explanations are now available in:
    # pipeline.forecasts - Dictionary mapping product_id to forecast DataFrame
    # pipeline.explanations - Dictionary mapping product_id to explanation object
    
    # You can access them like this:
    # forecasts = pipeline.forecasts
    # all_explanations = pipeline.explanations