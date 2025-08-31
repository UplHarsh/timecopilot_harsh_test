"""
Bitcoin Price Forecasting with TimeCopilot

This script shows how TimeCopilot handles highly volatile cryptocurrency data.
We'll download real Bitcoin prices and see which models work best for crypto forecasting.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from timecopilot import TimeCopilot
import warnings
warnings.filterwarnings('ignore')

# Set up clean plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def fetch_bitcoin_data():
    """Download Bitcoin data from Yahoo Finance"""
    print("Downloading Bitcoin price data...")
    
    # Get 3 years of Bitcoin data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    btc_data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    
    # Format for TimeCopilot
    df = pd.DataFrame({
        'unique_id': 'BTC-USD',
        'ds': btc_data.index,
        'y': btc_data['Close'].values
    })
    
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.reset_index(drop=True)
    
    print(f"Got {len(df)} days of Bitcoin data")
    print(f"Price range: ${df['y'].min():,.0f} - ${df['y'].max():,.0f}")
    
    return df

def visualize_data(df):
    """Show some basic charts of the Bitcoin data"""
    print("Creating data visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Bitcoin Price Analysis', fontsize=16, fontweight='bold')
    
    # Price over time
    axes[0,0].plot(df['ds'], df['y'], linewidth=1, alpha=0.8)
    axes[0,0].set_title('Bitcoin Price Over Time')
    axes[0,0].set_ylabel('Price (USD)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Price distribution
    axes[0,1].hist(df['y'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Price Distribution')
    axes[0,1].set_xlabel('Price (USD)')
    axes[0,1].set_ylabel('Frequency')
    
    # Daily returns
    df['returns'] = df['y'].pct_change()
    axes[1,0].plot(df['ds'], df['returns'], linewidth=0.5, alpha=0.7)
    axes[1,0].set_title('Daily Returns')
    axes[1,0].set_ylabel('Return (%)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Rolling volatility
    df['volatility'] = df['returns'].rolling(30).std()
    axes[1,1].plot(df['ds'], df['volatility'], linewidth=1, alpha=0.8)
    axes[1,1].set_title('30-Day Rolling Volatility')
    axes[1,1].set_ylabel('Volatility')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('weekend_project/bitcoin_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_intelligent_forecasting(df):
    """See how TimeCopilot handles volatile crypto data"""
    print("\n" + "="*50)
    print("INTELLIGENT FORECASTING ENGINE")
    print("="*50)
    
    # Set up TimeCopilot
    print("Setting up TimeCopilot...")
    tc = TimeCopilot(
        llm="openai:gpt-4o-mini",  # Using mini for speed
        retries=3
    )
    
    # Ask TimeCopilot to analyze the data
    forecast_query = """
    Look at this Bitcoin price data and forecast the next 30 days. 
    I'm particularly interested in:
    1. Which model handles cryptocurrency volatility best?
    2. What patterns do you see in recent price movements?
    3. How reliable is the forecast given Bitcoin's volatility?
    Please explain your reasoning for the model choice.
    """
    
    print("Running forecasting analysis...")
    print("This might take a minute as TimeCopilot tests different models...")
    
    # Generate forecast
    result = tc.forecast(
        df=df,
        h=30,  # 30 days ahead
        freq="D",  # Daily data
        query=forecast_query
    )
    
    print("\nForecasting complete!")
    print("\nRESULTS:")
    print("-" * 30)
    
    # Show results
    if hasattr(result, 'output') and result.output:
        print(f"Best Model: {result.output.selected_model}")
        print(f"Model Performance: {result.output.model_explanation}")
        print(f"Forecast Summary: {result.output.forecast_summary}")
    
    return result, tc

def run_genai_reasoning(tc, result):
    """Ask TimeCopilot questions about the forecast"""
    print("\n" + "="*40)
    print("AI REASONING LAYER")
    print("="*40)
    
    # Some interesting questions to ask
    questions = [
        "What are the biggest risks in this Bitcoin forecast?",
        "Which model performed best and why is it good for crypto data?",
        "What are the main uncertainties in this 30-day forecast?",
        "How does the recent trend compare to Bitcoin's historical patterns?",
        "What would you recommend for someone thinking about buying Bitcoin?"
    ]
    
    insights = {}
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 50)
        
        try:
            answer = tc.query(question)
            insights[question] = answer.output
            print(f"TimeCopilot: {answer.output}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            insights[question] = "Could not generate answer"
    
    return insights

def create_forecast_visualization(result):
    """Show the forecast results in a clean chart"""
    print("\nCreating forecast visualization...")
    
    if hasattr(result, 'fcst_df') and result.fcst_df is not None:
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Show recent history + forecast
        historical = result.fcst_df[result.fcst_df['ds'] <= result.fcst_df['ds'].max() - pd.Timedelta(days=30)]
        forecast = result.fcst_df[result.fcst_df['ds'] > result.fcst_df['ds'].max() - pd.Timedelta(days=30)]
        
        # Plot historical prices
        if len(historical) > 0:
            ax.plot(historical['ds'], historical['y'], 
                   label='Historical Prices', linewidth=2, color='navy', alpha=0.8)
        
        # Plot forecast
        if len(forecast) > 0 and hasattr(result, 'output'):
            model_name = result.output.selected_model
            if model_name in forecast.columns:
                ax.plot(forecast['ds'], forecast[model_name], 
                       label='Forecast', linewidth=2, color='red', linestyle='--')
                
                # Add confidence bands if available
                lo_cols = [col for col in forecast.columns if 'lo-' in col]
                hi_cols = [col for col in forecast.columns if 'hi-' in col]
                
                if lo_cols and hi_cols:
                    ax.fill_between(forecast['ds'], 
                                   forecast[lo_cols[0]], 
                                   forecast[hi_cols[0]],
                                   alpha=0.3, color='red', label='Confidence Band')
        
        ax.set_title('Bitcoin Price Forecast', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('weekend_project/bitcoin_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Forecast chart saved!")
    else:
        print("No forecast data available for visualization")

def main():
    """Run the complete Bitcoin analysis"""
    print("BITCOIN FORECASTING WITH TIMECOPILOT")
    print("="*50)
    
    try:
        # Step 1: Get the data
        df = fetch_bitcoin_data()
        visualize_data(df)
        
        # Step 2: Run forecasting
        result, tc = run_intelligent_forecasting(df)
        
        # Step 3: Ask questions
        insights = run_genai_reasoning(tc, result)
        
        # Step 4: Show forecast chart
        create_forecast_visualization(result)
        
        # Summary
        print("\nPROJECT SUMMARY")
        print("="*30)
        print("What we accomplished:")
        print("- Downloaded real Bitcoin data from Yahoo Finance")
        print("- Let TimeCopilot automatically select the best model")
        print("- Generated natural language insights about the forecast")
        print("- Created professional forecast visualizations")
        print("- Asked intelligent questions about the results")
        
        print(f"\nGenerated {len(insights)} AI insights")
        print("Charts saved: bitcoin_analysis.png, bitcoin_forecast.png")
        
        return result, insights
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure you have:")
        print("- OpenAI API key set in environment")
        print("- Internet connection for data download")
        print("- All packages installed (pip install -r requirements.txt)")
        return None, None

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
    
    result, insights = main()
