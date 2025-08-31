"""
E-commerce Sales Forecasting with TimeCopilot
==============================================

This script demonstrates TimeCopilot's capabilities on business data:
- Promotional effects and irregular spikes
- Growth trends and seasonal business patterns  
- Foundation model performance on retail data
- Business decision support through AI reasoning
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
sns.set_palette("coolwarm")

def generate_ecommerce_sales_data():
    """Generate realistic e-commerce sales data with business patterns"""
    print("🛒 Generating e-commerce sales data with business patterns...")
    
    # Create 3 years of daily data
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='D')[:-1]
    
    np.random.seed(123)  # For reproducible results
    
    # Base sales (growing business)
    base_sales = 10000
    growth_rate = 0.0003  # Daily growth rate
    trend = base_sales * (1 + growth_rate * np.arange(len(dates)))
    
    # Seasonal patterns
    # Holiday seasonality (Q4 boost)
    yearly_pattern = 3000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25 + 4.5)
    
    # Weekly pattern (weekend boost for retail)
    weekly_pattern = 1000 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    
    # Monthly payday effect (boost around 1st and 15th)
    monthly_pattern = 500 * (np.sin(2 * np.pi * np.arange(len(dates)) / 30.4) + 
                           np.sin(4 * np.pi * np.arange(len(dates)) / 30.4))
    
    # Create base sales
    sales = trend + yearly_pattern + weekly_pattern + monthly_pattern
    
    # Add promotional events and special days
    promo_dates = []
    promo_effects = []
    
    # Black Friday/Cyber Monday (4x sales)
    for year in range(2021, 2024):
        bf_date = datetime(year, 11, 25)  # Approximate Black Friday
        if bf_date <= end_date:
            bf_idx = (bf_date - start_date).days
            if 0 <= bf_idx < len(dates):
                promo_dates.append(bf_idx)
                promo_effects.append(3.0)  # 4x sales
                # Cyber Monday
                cm_idx = bf_idx + 3
                if cm_idx < len(dates):
                    promo_dates.append(cm_idx)
                    promo_effects.append(2.5)  # 3.5x sales
    
    # Valentine's Day (2x sales)
    for year in range(2021, 2024):
        vd_date = datetime(year, 2, 14)
        if vd_date <= end_date:
            vd_idx = (vd_date - start_date).days
            if 0 <= vd_idx < len(dates):
                promo_dates.append(vd_idx)
                promo_effects.append(1.5)  # 2.5x sales
    
    # Mother's Day (1.5x sales)
    mothers_days = [datetime(2021, 5, 9), datetime(2022, 5, 8), datetime(2023, 5, 14)]
    for md_date in mothers_days:
        if md_date <= end_date:
            md_idx = (md_date - start_date).days
            if 0 <= md_idx < len(dates):
                promo_dates.append(md_idx)
                promo_effects.append(1.0)  # 2x sales
    
    # Apply promotional effects
    for idx, effect in zip(promo_dates, promo_effects):
        sales[idx] *= (1 + effect)
    
    # Add flash sales (random promotional spikes)
    for _ in range(50):  # 50 flash sales over 3 years
        flash_day = np.random.randint(0, len(dates))
        flash_effect = np.random.uniform(0.5, 2.0)  # 1.5x to 3x sales
        sales[flash_day] *= (1 + flash_effect)
    
    # Add random noise
    noise = np.random.normal(0, sales * 0.1)  # 10% noise
    sales += noise
    
    # Ensure non-negative sales
    sales = np.maximum(sales, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'unique_id': 'ecommerce_sales',
        'ds': dates,
        'y': sales
    })
    
    # Add business features for analysis
    df['month'] = df['ds'].dt.month
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['quarter'] = df['ds'].dt.quarter
    
    # Mark promotional periods
    df['is_promo'] = False
    for idx in promo_dates:
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('is_promo')] = True
    
    print(f"✅ Generated {len(df)} days of e-commerce sales data")
    print(f"💰 Sales range: ${df['y'].min():,.0f} - ${df['y'].max():,.0f}")
    print(f"📈 Growth trend: {((df['y'].iloc[-365:].mean() / df['y'].iloc[:365].mean() - 1) * 100):.1f}% year-over-year")
    
    return df

def visualize_business_patterns(df):
    """Create business-focused visualizations"""
    print("📊 Analyzing e-commerce sales patterns...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('E-commerce Sales Analysis - Business Intelligence', fontsize=16, fontweight='bold')
    
    # Sales trend over time
    axes[0,0].plot(df['ds'], df['y'], linewidth=0.8, alpha=0.7)
    axes[0,0].set_title('Sales Trend Over Time')
    axes[0,0].set_ylabel('Daily Sales ($)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Add trend line
    z = np.polyfit(range(len(df)), df['y'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['ds'], p(range(len(df))), "r--", alpha=0.8, linewidth=2)
    
    # Seasonal pattern (monthly)
    monthly_sales = df.groupby('month')['y'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[0,1].bar(range(1, 13), monthly_sales.values, color='steelblue', alpha=0.7)
    axes[0,1].set_title('Average Sales by Month')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Average Daily Sales ($)')
    axes[0,1].set_xticks(range(1, 13))
    axes[0,1].set_xticklabels(month_names)
    
    # Weekend vs weekday performance
    weekend_comparison = df.groupby('is_weekend')['y'].mean()
    axes[0,2].bar(['Weekday', 'Weekend'], weekend_comparison.values, 
                  color=['orange', 'green'], alpha=0.7)
    axes[0,2].set_title('Weekday vs Weekend Sales')
    axes[0,2].set_ylabel('Average Daily Sales ($)')
    
    # Promotional impact
    promo_comparison = df.groupby('is_promo')['y'].mean()
    axes[1,0].bar(['Regular Days', 'Promotional Days'], promo_comparison.values,
                  color=['lightblue', 'red'], alpha=0.7)
    axes[1,0].set_title('Promotional Impact on Sales')
    axes[1,0].set_ylabel('Average Daily Sales ($)')
    
    # Sales distribution
    axes[1,1].hist(df['y'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_title('Sales Distribution')
    axes[1,1].set_xlabel('Daily Sales ($)')
    axes[1,1].set_ylabel('Frequency')
    
    # Quarterly performance
    quarterly_sales = df.groupby(['ds'].map(lambda x: x.year), 'quarter')['y'].sum().unstack()
    quarterly_sales.plot(kind='bar', ax=axes[1,2], alpha=0.7)
    axes[1,2].set_title('Quarterly Sales Performance')
    axes[1,2].set_xlabel('Year')
    axes[1,2].set_ylabel('Quarterly Sales ($)')
    axes[1,2].legend(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('weekend_project/ecommerce_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_business_forecasting(df):
    """Demonstrate business-focused forecasting"""
    print("\n🤖 INTELLIGENT FORECASTING ENGINE - BUSINESS FORECASTING")
    print("="*60)
    
    # Use last 18 months for training (business cycles)
    train_df = df.tail(18*30).copy().reset_index(drop=True)
    
    print("🔧 Initializing TimeCopilot for business forecasting...")
    tc = TimeCopilot(
        llm="openai:gpt-4o-mini",
        retries=3
    )
    
    # Business forecasting query
    forecast_query = """
    Analyze this e-commerce sales data and forecast the next 60 days for business planning.
    
    This data contains:
    1. Strong growth trend (expanding business)
    2. Seasonal patterns (Q4 holiday boost, monthly pay cycles)
    3. Promotional spikes (Black Friday, Valentine's Day, flash sales)
    4. Weekend vs weekday differences
    
    Focus on:
    - Which model best captures both trend and promotional effects?
    - How accurately can we predict promotional lift?
    - What's the reliability for inventory planning decisions?
    - How do foundation models handle business seasonality vs statistical models?
    - What are the revenue forecasting confidence intervals?
    
    This forecast will drive:
    - Inventory procurement decisions ($500K budget)
    - Marketing campaign planning
    - Staff scheduling for customer service
    - Revenue guidance for investors
    """
    
    print("🎯 Running business forecasting analysis...")
    print(f"📊 Training data: {len(train_df)} days ({len(train_df)/30:.1f} months)")
    
    # Generate forecast
    result = tc.forecast(
        df=train_df,
        h=60,  # 60 days forecast
        freq="D",   # Daily frequency
        query=forecast_query
    )
    
    print("\n✅ Business forecasting completed!")
    print("\n📊 BUSINESS FORECASTING RESULTS:")
    print("-" * 50)
    
    if hasattr(result, 'output') and result.output:
        print(f"🎯 Best Model: {result.output.selected_model}")
        print(f"💼 Business Insights: {result.output.forecast_summary}")
        print(f"📈 Model Performance: {result.output.model_comparison}")
        
    return result, tc, train_df

def run_business_reasoning(tc):
    """Demonstrate AI reasoning for business decisions"""
    print("\n🧠 GENAI REASONING LAYER - BUSINESS INTELLIGENCE")
    print("="*55)
    
    business_questions = [
        "What's the expected revenue for the next 60 days and what's the confidence interval?",
        "Which days are likely to have the highest sales and why?",
        "How should we adjust inventory levels based on this forecast?",
        "What are the main business risks if the forecast is wrong?",
        "How do promotional effects impact forecast accuracy?",
        "What marketing strategies would you recommend based on these patterns?",
        "How reliable is this forecast for investor revenue guidance?",
        "What early warning signs should we monitor to detect forecast deviations?"
    ]
    
    insights = {}
    
    for i, question in enumerate(business_questions, 1):
        print(f"\n❓ Business Question {i}: {question}")
        print("-" * 70)
        
        try:
            answer = tc.query(question)
            insights[question] = answer.output
            print(f"🤖 Business AI Advisor: {answer.output}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            insights[question] = "Could not generate answer"
    
    return insights

def create_business_dashboard(result, train_df):
    """Create executive business dashboard"""
    print("\n📊 Creating business intelligence dashboard...")
    
    if hasattr(result, 'fcst_df') and result.fcst_df is not None:
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('E-commerce Sales Forecast - Executive Dashboard', 
                    fontsize=16, fontweight='bold')
        
        fcst_df = result.fcst_df
        
        # Revenue forecast with confidence bands
        historical_recent = train_df.tail(60)  # Last 60 days
        forecast = fcst_df.tail(60)  # Next 60 days
        
        axes[0,0].plot(historical_recent['ds'], historical_recent['y'],
                      label='Historical Sales', linewidth=2, color='blue', alpha=0.8)
        
        if result.output and result.output.selected_model in forecast.columns:
            axes[0,0].plot(forecast['ds'], forecast[result.output.selected_model],
                          label='Forecast', linewidth=2, color='red', linestyle='--')
            
            # Add confidence intervals if available
            model_cols = forecast.columns.tolist()
            lo_cols = [col for col in model_cols if 'lo-' in col and result.output.selected_model.replace('_', '') in col]
            hi_cols = [col for col in model_cols if 'hi-' in col and result.output.selected_model.replace('_', '') in col]
            
            if lo_cols and hi_cols:
                axes[0,0].fill_between(forecast['ds'], 
                                     forecast[lo_cols[0]], 
                                     forecast[hi_cols[0]],
                                     alpha=0.3, color='red', label='Confidence Interval')
        
        axes[0,0].set_title('60-Day Revenue Forecast')
        axes[0,0].set_ylabel('Daily Sales ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Weekly revenue projections
        if 'ds' in forecast.columns and result.output and result.output.selected_model in forecast.columns:
            forecast['week'] = forecast['ds'].dt.isocalendar().week
            weekly_forecast = forecast.groupby('week')[result.output.selected_model].sum()
            
            axes[0,1].bar(range(len(weekly_forecast)), weekly_forecast.values, 
                         alpha=0.7, color='green')
            axes[0,1].set_title('Weekly Revenue Projections')
            axes[0,1].set_xlabel('Week Number')
            axes[0,1].set_ylabel('Weekly Sales ($)')
            axes[0,1].grid(True, alpha=0.3)
        
        # Revenue growth analysis
        if result.output and result.output.selected_model in forecast.columns:
            historical_avg = train_df.tail(30)['y'].mean()
            forecast_avg = forecast[result.output.selected_model].mean()
            growth_rate = ((forecast_avg / historical_avg) - 1) * 100
            
            axes[1,0].bar(['Last 30 Days\nAvg', 'Next 60 Days\nForecast Avg'], 
                         [historical_avg, forecast_avg],
                         color=['blue', 'red'], alpha=0.7)
            axes[1,0].set_title(f'Revenue Growth Analysis\n({growth_rate:+.1f}% projected change)')
            axes[1,0].set_ylabel('Average Daily Sales ($)')
            
            # Add growth rate annotation
            mid_point = (historical_avg + forecast_avg) / 2
            axes[1,0].annotate(f'{growth_rate:+.1f}%', xy=(0.5, mid_point), 
                             ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Model performance for business confidence
        if hasattr(result, 'eval_df') and result.eval_df is not None:
            eval_df = result.eval_df
            if not eval_df.empty:
                model_cols = [col for col in eval_df.columns if col not in ['metric']]
                if model_cols and len(eval_df) > 0:
                    mase_data = eval_df[eval_df['metric'] == 'MASE'].iloc[0] if 'MASE' in eval_df['metric'].values else None
                    
                    if mase_data is not None:
                        scores = [mase_data[col] for col in model_cols if col in mase_data.index]
                        colors = ['gold' if col == result.output.selected_model else 'lightblue' 
                                for col in model_cols]
                        
                        axes[1,1].bar(range(len(model_cols)), scores, color=colors, alpha=0.7)
                        axes[1,1].set_title('Model Performance\n(MASE - Lower = Better)')
                        axes[1,1].set_ylabel('MASE Score')
                        axes[1,1].set_xticks(range(len(model_cols)))
                        axes[1,1].set_xticklabels(model_cols, rotation=45)
                        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('weekend_project/ecommerce_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate key business metrics
        if result.output and result.output.selected_model in forecast.columns:
            total_forecast_revenue = forecast[result.output.selected_model].sum()
            avg_daily_revenue = forecast[result.output.selected_model].mean()
            max_daily_revenue = forecast[result.output.selected_model].max()
            
            print(f"\n💰 KEY BUSINESS METRICS:")
            print(f"   • Total 60-day revenue forecast: ${total_forecast_revenue:,.0f}")
            print(f"   • Average daily revenue: ${avg_daily_revenue:,.0f}")
            print(f"   • Peak day revenue: ${max_daily_revenue:,.0f}")
            print(f"   • Growth vs last 30 days: {growth_rate:+.1f}%")
        
        print("✅ Executive dashboard generated!")
    else:
        print("⚠️ No forecast data available for dashboard")

def main():
    """Main execution function for e-commerce forecasting"""
    print("🛒 E-COMMERCE SALES FORECASTING WITH TIMECOPILOT")
    print("="*55)
    
    try:
        # Step 1: Generate and analyze business data
        df = generate_ecommerce_sales_data()
        visualize_business_patterns(df)
        
        # Step 2: Business-focused forecasting
        result, tc, train_df = run_business_forecasting(df)
        
        # Step 3: Business intelligence reasoning
        insights = run_business_reasoning(tc)
        
        # Step 4: Create executive dashboard
        create_business_dashboard(result, train_df)
        
        # Step 5: Executive summary
        print("\n📋 EXECUTIVE SUMMARY")
        print("="*35)
        print("✅ Business pattern recognition completed")
        print("✅ 60-day revenue forecast generated")
        print("✅ Promotional impact analysis completed")
        print("✅ Growth trend projection provided")
        print("✅ Business risk assessment included")
        print("✅ Investment guidance insights delivered")
        
        print(f"\n🎯 Business Insights: {len(insights)} strategic recommendations")
        print("📊 Dashboards: ecommerce_patterns.png, ecommerce_dashboard.png")
        print("💼 Ready for business planning and investor presentations!")
        
        return result, insights
        
    except Exception as e:
        print(f"\n❌ Error during business analysis: {str(e)}")
        return None, None

if __name__ == "__main__":
    result, insights = main()
