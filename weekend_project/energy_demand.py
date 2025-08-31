"""
Energy Demand Forecasting with TimeCopilot
==========================================

This script demonstrates TimeCopilot's capabilities on energy consumption data:
- Multi-seasonal pattern recognition (daily + weekly cycles)
- Foundation model performance on utility data
- Complex seasonality handling
- Business-critical forecasting accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from timecopilot import TimeCopilot
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

def generate_synthetic_energy_data():
    """Generate realistic synthetic energy consumption data"""
    print("⚡ Generating synthetic energy consumption data...")
    
    # Create 2 years of hourly data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='H')[:-1]  # Remove last to get exactly 2 years
    
    np.random.seed(42)  # For reproducible results
    
    # Base consumption (MW)
    base_consumption = 1000
    
    # Daily seasonality (peak during day, low at night)
    daily_pattern = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 24 - np.pi/2)
    
    # Weekly seasonality (lower on weekends)
    weekly_pattern = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*7))
    
    # Seasonal pattern (higher in summer/winter for AC/heating)
    yearly_pattern = 150 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365.25))
    
    # Temperature effect (simplified)
    temp_effect = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365.25) + np.pi)
    
    # Random noise and spikes
    noise = np.random.normal(0, 30, len(dates))
    
    # Create consumption with all patterns
    consumption = (base_consumption + 
                  daily_pattern + 
                  weekly_pattern + 
                  yearly_pattern + 
                  temp_effect + 
                  noise)
    
    # Add some random outages and peak events
    for _ in range(10):
        outage_start = np.random.randint(0, len(dates)-48)
        consumption[outage_start:outage_start+np.random.randint(1,6)] *= 0.3  # Outage
        
    for _ in range(20):
        peak_hour = np.random.randint(0, len(dates))
        consumption[peak_hour] *= 1.5  # Peak event
    
    # Ensure non-negative values
    consumption = np.maximum(consumption, 100)
    
    # Create DataFrame for TimeCopilot
    df = pd.DataFrame({
        'unique_id': 'energy_demand',
        'ds': dates,
        'y': consumption
    })
    
    # Add time-based features for analysis
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    
    print(f"✅ Generated {len(df)} hours of energy consumption data")
    print(f"⚡ Demand range: {df['y'].min():.0f} - {df['y'].max():.0f} MW")
    
    return df

def visualize_energy_patterns(df):
    """Create comprehensive visualizations of energy patterns"""
    print("📊 Analyzing energy consumption patterns...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Energy Demand Analysis - Pattern Recognition', fontsize=16, fontweight='bold')
    
    # Overall time series (last 90 days)
    recent_data = df.tail(90*24)
    axes[0,0].plot(recent_data['ds'], recent_data['y'], linewidth=0.8, alpha=0.8)
    axes[0,0].set_title('Energy Demand - Last 90 Days')
    axes[0,0].set_ylabel('Demand (MW)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Daily pattern
    daily_avg = df.groupby('hour')['y'].mean()
    axes[0,1].plot(daily_avg.index, daily_avg.values, marker='o', linewidth=2)
    axes[0,1].set_title('Average Hourly Pattern')
    axes[0,1].set_xlabel('Hour of Day')
    axes[0,1].set_ylabel('Average Demand (MW)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Weekly pattern
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_avg = df.groupby('day_of_week')['y'].mean()
    axes[0,2].bar(range(7), weekly_avg.values, color='skyblue', alpha=0.7)
    axes[0,2].set_title('Average Demand by Day of Week')
    axes[0,2].set_xlabel('Day of Week')
    axes[0,2].set_ylabel('Average Demand (MW)')
    axes[0,2].set_xticks(range(7))
    axes[0,2].set_xticklabels(day_names)
    
    # Monthly pattern
    monthly_avg = df.groupby('month')['y'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1,0].bar(range(1, 13), monthly_avg.values, color='lightcoral', alpha=0.7)
    axes[1,0].set_title('Average Demand by Month')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Average Demand (MW)')
    axes[1,0].set_xticks(range(1, 13))
    axes[1,0].set_xticklabels(month_names)
    
    # Demand distribution
    axes[1,1].hist(df['y'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].set_title('Demand Distribution')
    axes[1,1].set_xlabel('Demand (MW)')
    axes[1,1].set_ylabel('Frequency')
    
    # Heatmap: Hour vs Day of Week
    pivot_data = df.pivot_table(values='y', index='hour', columns='day_of_week', aggfunc='mean')
    im = axes[1,2].imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    axes[1,2].set_title('Demand Heatmap: Hour vs Day')
    axes[1,2].set_xlabel('Day of Week (0=Monday)')
    axes[1,2].set_ylabel('Hour of Day')
    axes[1,2].set_xticks(range(7))
    axes[1,2].set_xticklabels(day_names)
    plt.colorbar(im, ax=axes[1,2], label='Demand (MW)')
    
    plt.tight_layout()
    plt.savefig('weekend_project/energy_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_intelligent_forecasting(df):
    """Demonstrate TimeCopilot's handling of complex seasonality"""
    print("\n🤖 INTELLIGENT FORECASTING ENGINE - COMPLEX SEASONALITY")
    print("="*60)
    
    # Take recent data for training (last 6 months)
    train_df = df.tail(6*30*24).copy().reset_index(drop=True)
    
    print("🔧 Initializing TimeCopilot for multi-seasonal forecasting...")
    tc = TimeCopilot(
        llm="openai:gpt-4o-mini",
        retries=3
    )
    
    # Complex forecasting query
    forecast_query = """
    Analyze this hourly energy demand data and forecast the next 7 days (168 hours).
    This data has multiple seasonal patterns:
    1. Daily cycles (peak during business hours, low at night)
    2. Weekly cycles (lower demand on weekends)  
    3. Monthly/seasonal patterns (weather-dependent)
    
    Focus on:
    - Which model best captures multiple seasonal patterns?
    - How accurately can we predict peak demand hours?
    - What are the reliability concerns for grid operations?
    - How do foundation models perform vs traditional time series models?
    
    This forecast is critical for electricity grid planning and operations.
    """
    
    print("🎯 Running multi-seasonal forecasting...")
    print(f"📊 Training data: {len(train_df)} hours ({len(train_df)//24:.1f} days)")
    
    # Generate forecast
    result = tc.forecast(
        df=train_df,
        h=168,  # 7 days * 24 hours
        freq="H",   # Hourly frequency
        query=forecast_query
    )
    
    print("\n✅ Forecasting completed!")
    print("\n📊 FORECASTING RESULTS:")
    print("-" * 40)
    
    if hasattr(result, 'output') and result.output:
        print(f"🎯 Selected Model: {result.output.selected_model}")
        print(f"📈 Cross-validation Performance: {result.output.model_comparison}")
        print(f"🔍 Technical Analysis: {result.output.technical_details}")
        
    return result, tc, train_df

def run_genai_reasoning(tc):
    """Demonstrate AI reasoning about energy patterns"""
    print("\n🧠 GENAI REASONING LAYER - ENERGY INSIGHTS")
    print("="*50)
    
    energy_questions = [
        "What are the most critical hours for energy demand forecasting accuracy?",
        "How do the seasonal patterns interact in this energy consumption data?", 
        "Which forecasting model is most suitable for grid operations and why?",
        "What are the main risks if we underestimate peak demand?",
        "How reliable is this 7-day forecast for electricity grid planning?",
        "What patterns indicate potential grid stress periods?",
        "How does weekend vs weekday demand differ and why is this important?"
    ]
    
    insights = {}
    
    for i, question in enumerate(energy_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 60)
        
        try:
            answer = tc.query(question)
            insights[question] = answer.output
            print(f"🤖 Energy Expert AI: {answer.output}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            insights[question] = "Could not generate answer"
    
    return insights

def create_energy_forecast_viz(result, train_df):
    """Create professional energy forecast visualizations"""
    print("\n📊 Creating energy forecast visualizations...")
    
    if hasattr(result, 'fcst_df') and result.fcst_df is not None:
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Energy Demand Forecast - Grid Operations Dashboard', 
                    fontsize=16, fontweight='bold')
        
        fcst_df = result.fcst_df
        
        # Main forecast plot
        # Show last 7 days of historical + 7 days forecast
        historical_cutoff = len(train_df) - 7*24  # Last 7 days of training
        historical = train_df.iloc[historical_cutoff:].copy()
        forecast = fcst_df.tail(168).copy()  # Last 168 hours (7 days)
        
        axes[0,0].plot(historical['ds'], historical['y'], 
                      label='Historical Demand', linewidth=2, color='blue', alpha=0.8)
        
        if result.output and result.output.selected_model in forecast.columns:
            axes[0,0].plot(forecast['ds'], forecast[result.output.selected_model],
                          label='Forecasted Demand', linewidth=2, color='red', linestyle='--')
        
        axes[0,0].set_title('7-Day Energy Demand Forecast')
        axes[0,0].set_ylabel('Demand (MW)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Daily pattern comparison
        if 'ds' in forecast.columns:
            forecast['hour'] = forecast['ds'].dt.hour
            if result.output and result.output.selected_model in forecast.columns:
                daily_forecast = forecast.groupby('hour')[result.output.selected_model].mean()
                daily_historical = train_df.groupby('hour')['y'].mean()
                
                axes[0,1].plot(daily_historical.index, daily_historical.values, 
                              'o-', label='Historical Pattern', linewidth=2)
                axes[0,1].plot(daily_forecast.index, daily_forecast.values, 
                              's--', label='Forecast Pattern', linewidth=2)
                axes[0,1].set_title('Daily Demand Pattern Comparison')
                axes[0,1].set_xlabel('Hour of Day')
                axes[0,1].set_ylabel('Average Demand (MW)')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
        
        # Peak demand analysis
        if result.output and result.output.selected_model in forecast.columns:
            forecast_vals = forecast[result.output.selected_model]
            peak_hours = forecast[forecast_vals == forecast_vals.max()]
            
            axes[1,0].bar(['Historical Max', 'Forecast Max'], 
                         [train_df['y'].max(), forecast_vals.max()],
                         color=['blue', 'red'], alpha=0.7)
            axes[1,0].set_title('Peak Demand Comparison')
            axes[1,0].set_ylabel('Peak Demand (MW)')
            
            # Add peak hour info
            if not peak_hours.empty and 'ds' in peak_hours.columns:
                peak_time = peak_hours.iloc[0]['ds']
                axes[1,0].text(1, forecast_vals.max() + 10, 
                              f'Peak: {peak_time.strftime("%Y-%m-%d %H:%M")}',
                              ha='center', fontsize=10)
        
        # Model performance visualization
        if hasattr(result, 'eval_df') and result.eval_df is not None:
            eval_df = result.eval_df
            if not eval_df.empty and 'metric' in eval_df.columns:
                model_cols = [col for col in eval_df.columns if col not in ['metric']]
                if model_cols:
                    # Show MASE scores
                    mase_row = eval_df[eval_df['metric'] == 'MASE'].iloc[0] if len(eval_df) > 0 else None
                    if mase_row is not None:
                        scores = [mase_row[col] for col in model_cols if col in mase_row.index]
                        axes[1,1].bar(range(len(model_cols)), scores, alpha=0.7)
                        axes[1,1].set_title('Model Performance (MASE - Lower is Better)')
                        axes[1,1].set_ylabel('MASE Score')
                        axes[1,1].set_xticks(range(len(model_cols)))
                        axes[1,1].set_xticklabels(model_cols, rotation=45)
                        
                        # Highlight best model
                        if result.output and result.output.selected_model in model_cols:
                            best_idx = model_cols.index(result.output.selected_model)
                            axes[1,1].bar(best_idx, scores[best_idx], color='gold', alpha=0.9)
        
        plt.tight_layout()
        plt.savefig('weekend_project/energy_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Energy forecast dashboard saved!")
    else:
        print("⚠️ No forecast data available for visualization")

def main():
    """Main execution function for energy demand forecasting"""
    print("⚡ ENERGY DEMAND FORECASTING WITH TIMECOPILOT")
    print("="*55)
    
    try:
        # Step 1: Generate and analyze synthetic energy data
        df = generate_synthetic_energy_data()
        visualize_energy_patterns(df)
        
        # Step 2: Intelligent multi-seasonal forecasting
        result, tc, train_df = run_intelligent_forecasting(df)
        
        # Step 3: GenAI reasoning about energy patterns
        insights = run_genai_reasoning(tc)
        
        # Step 4: Create grid operations dashboard
        create_energy_forecast_viz(result, train_df)
        
        # Step 5: Summary for grid operators
        print("\n📋 GRID OPERATIONS SUMMARY")
        print("="*40)
        print("✅ Multi-seasonal pattern recognition completed")
        print("✅ 7-day demand forecast generated")
        print("✅ Peak demand periods identified") 
        print("✅ Model reliability assessment provided")
        print("✅ Grid planning insights generated")
        
        print(f"\n🎯 AI Insights Generated: {len(insights)} expert responses")
        print("📊 Visualizations: energy_patterns.png, energy_forecast.png")
        print("⚡ Ready for grid operations planning!")
        
        return result, insights
        
    except Exception as e:
        print(f"\n❌ Error during energy analysis: {str(e)}")
        return None, None

if __name__ == "__main__":
    result, insights = main()
