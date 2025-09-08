#!/usr/bin/env python3
"""
Test script for the TimeCopilot forecasting and explanation functionality.
This script replicates the problematic code pattern from the user's issue
but with the proper implementation.
"""

import pandas as pd
import sys
import os

# Add the timecopilot module to the path
sys.path.insert(0, '/home/runner/work/timecopilot_harsh_test/timecopilot_harsh_test')

def test_forecasting_with_explanations():
    """
    Test the complete forecasting and explanation workflow.
    This replicates the user's code pattern but with proper implementation.
    """
    print("Testing TimeCopilot Forecasting with Explanations")
    print("=" * 60)
    
    try:
        # Import the necessary modules
        from timecopilot import TimeCopilotForecaster, ForecastExplainer
        from timecopilot.models.stats import AutoARIMA, AutoETS, SeasonalNaive
        
        print("✓ Successfully imported TimeCopilot modules")
        
        # Load sample data
        print("\nLoading sample data...")
        try:
            df = pd.read_csv(
                "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
                parse_dates=["ds"]
            )
            
            # Create multiple products for testing
            df_products = []
            
            # Original data
            original = df.copy()
            original['unique_id'] = 'AirPassengers'
            df_products.append(original)
            
            # Product A (modified)
            product_a = df.copy()
            product_a['unique_id'] = 'Product_A'
            product_a['y'] = product_a['y'] * 1.2
            df_products.append(product_a)
            
            # Product B (modified)
            product_b = df.copy()
            product_b['unique_id'] = 'Product_B'
            product_b['y'] = product_b['y'] * 0.8 + 50
            df_products.append(product_b)
            
            df = pd.concat(df_products, ignore_index=True)
            print(f"✓ Loaded data for {df['unique_id'].nunique()} products")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
        
        # Initialize forecaster
        print("\nInitializing forecaster...")
        try:
            tcf = TimeCopilotForecaster(
                models=[
                    AutoARIMA(),
                    AutoETS(), 
                    SeasonalNaive(),
                ]
            )
            print("✓ Successfully initialized TimeCopilotForecaster")
        except Exception as e:
            print(f"✗ Error initializing forecaster: {e}")
            return False
        
        # Generate forecasts
        print("\nGenerating forecasts...")
        try:
            forecast_df = tcf.forecast(df=df, h=12, level=[80, 90])
            print(f"✓ Generated forecasts: {len(forecast_df)} rows")
            
            # Split forecasts by product_id (this is what was missing in the original code)
            forecasts = {}
            for product_id in df['unique_id'].unique():
                product_forecast = forecast_df[forecast_df['unique_id'] == product_id].copy()
                if not product_forecast.empty:
                    forecasts[product_id] = product_forecast
            
            print(f"✓ Split forecasts for {len(forecasts)} products")
            
        except Exception as e:
            print(f"✗ Error generating forecasts: {e}")
            return False
        
        # Initialize explainer
        print("\nInitializing explainer...")
        try:
            explainer = ForecastExplainer()
            print("✓ Successfully initialized ForecastExplainer")
        except Exception as e:
            print(f"✗ Error initializing explainer: {e}")
            return False
        
        # Generate explanations (this is the code that was failing before)
        print("\nGenerating explanations...")
        try:
            all_explanations = {}
            for product_id in df['unique_id'].unique()[:3]:  # Limit to first 3 products
                if product_id in forecasts:  # This line was causing NameError before
                    explanation = explainer.generate_explanation(
                        df=df,
                        forecast_df=forecasts[product_id],
                        product_id=product_id
                    )
                    all_explanations[product_id] = explanation
                    print(f"✓ Generated explanation for {product_id}")
            
            print(f"✓ Generated {len(all_explanations)} explanations")
            
        except Exception as e:
            print(f"✗ Error generating explanations: {e}")
            return False
        
        # Display results
        print("\nResults Summary:")
        print("-" * 40)
        for product_id, explanation in all_explanations.items():
            print(f"\nProduct: {product_id}")
            data_insights = explanation.data_insights
            if 'data_points' in data_insights:
                print(f"  - Data points: {data_insights['data_points']}")
            if 'trend' in data_insights:
                print(f"  - Trend: {data_insights['trend']}")
            if 'mean_value' in data_insights:
                print(f"  - Average value: {data_insights['mean_value']:.2f}")
        
        print("\n" + "=" * 60)
        print("✓ TEST PASSED: All functionality working correctly!")
        print("✓ The 'forecasts' variable is now properly defined")
        print("✓ The explainer generates explanations successfully")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_pipeline():
    """
    Test the simplified pipeline approach.
    """
    print("\n\nTesting Simplified Pipeline")
    print("=" * 40)
    
    try:
        from timecopilot import TimeCopilotForecastingPipeline
        
        # Initialize pipeline
        pipeline = TimeCopilotForecastingPipeline()
        
        # Load data
        df = pipeline.load_sample_data()
        print(f"✓ Loaded data for {df['unique_id'].nunique()} products")
        
        # Generate forecasts
        forecasts = pipeline.generate_forecasts(df, h=6)  # Shorter horizon for testing
        print(f"✓ Generated forecasts for {len(forecasts)} products")
        
        # Generate explanations
        explanations = pipeline.generate_explanations(df, list(forecasts.keys())[:2])  # Limit to 2 products
        print(f"✓ Generated explanations for {len(explanations)} products")
        
        # The variables are now available as expected:
        # - pipeline.forecasts contains the forecasts dict
        # - pipeline.explanations contains the explanations dict
        
        print("\n✓ PIPELINE TEST PASSED: All functionality working!")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running TimeCopilot Tests")
    print("========================")
    
    # Test 1: Direct approach (replicating user's code pattern)
    success1 = test_forecasting_with_explanations()
    
    # Test 2: Pipeline approach (simplified usage)
    success2 = test_simple_pipeline()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("The solution fixes the original NameError and provides working functionality.")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)