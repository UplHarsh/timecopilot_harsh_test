"""
TimeCopilot Weekend Project - Utilities
=======================================

Common utilities and helper functions for the weekend project demos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def setup_plotting_style():
    """Set up consistent plotting style for all demos"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10

def print_section_header(title: str, char: str = "="):
    """Print a formatted section header"""
    width = max(50, len(title) + 10)
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection_header(title: str, char: str = "-"):
    """Print a formatted subsection header"""
    width = max(40, len(title) + 6)
    print(f"\n{title}")
    print(f"{char * width}")

def format_currency(value: float, decimals: int = 0) -> str:
    """Format a number as currency"""
    if decimals == 0:
        return f"${value:,.0f}"
    else:
        return f"${value:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a number as percentage"""
    return f"{value:+.{decimals}f}%"

def calculate_basic_stats(df: pd.DataFrame, value_col: str = 'y') -> Dict:
    """Calculate basic statistics for a time series"""
    stats = {
        'count': len(df),
        'mean': df[value_col].mean(),
        'std': df[value_col].std(),
        'min': df[value_col].min(),
        'max': df[value_col].max(),
        'range': df[value_col].max() - df[value_col].min(),
        'cv': df[value_col].std() / df[value_col].mean()  # Coefficient of variation
    }
    
    # Calculate growth rate if we have dates
    if 'ds' in df.columns:
        first_year_avg = df.head(min(365, len(df)//2))[value_col].mean()
        last_year_avg = df.tail(min(365, len(df)//2))[value_col].mean()
        stats['growth_rate'] = ((last_year_avg / first_year_avg) - 1) * 100
    
    return stats

def print_data_summary(df: pd.DataFrame, name: str, value_col: str = 'y'):
    """Print a formatted data summary"""
    stats = calculate_basic_stats(df, value_col)
    
    print(f"📊 {name} Dataset Summary:")
    print(f"   • Data points: {stats['count']:,}")
    print(f"   • Value range: {format_currency(stats['min'], 2)} - {format_currency(stats['max'], 2)}")
    print(f"   • Average: {format_currency(stats['mean'], 2)}")
    print(f"   • Volatility (CV): {stats['cv']:.2f}")
    
    if 'growth_rate' in stats:
        print(f"   • Growth rate: {format_percentage(stats['growth_rate'])}")

def create_forecast_comparison_plot(historical_df: pd.DataFrame, 
                                  forecast_df: pd.DataFrame,
                                  model_name: str,
                                  title: str,
                                  confidence_intervals: bool = True) -> plt.Figure:
    """Create a standard forecast comparison plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot historical data
    ax.plot(historical_df['ds'], historical_df['y'],
           label='Historical Data', linewidth=2, color='blue', alpha=0.8)
    
    # Plot forecast
    if model_name in forecast_df.columns:
        ax.plot(forecast_df['ds'], forecast_df[model_name],
               label='Forecast', linewidth=2, color='red', linestyle='--')
        
        # Add confidence intervals if available
        if confidence_intervals:
            lo_cols = [col for col in forecast_df.columns if 'lo-' in col and model_name.replace('_', '') in col]
            hi_cols = [col for col in forecast_df.columns if 'hi-' in col and model_name.replace('_', '') in col]
            
            if lo_cols and hi_cols:
                ax.fill_between(forecast_df['ds'], 
                               forecast_df[lo_cols[0]], 
                               forecast_df[hi_cols[0]],
                               alpha=0.3, color='red', label='Confidence Interval')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def extract_forecast_metrics(result, forecast_df: pd.DataFrame) -> Dict:
    """Extract key forecast metrics from TimeCopilot results"""
    metrics = {}
    
    if hasattr(result, 'output') and result.output:
        metrics['selected_model'] = result.output.selected_model
        
        if result.output.selected_model in forecast_df.columns:
            forecast_values = forecast_df[result.output.selected_model]
            metrics['forecast_mean'] = forecast_values.mean()
            metrics['forecast_std'] = forecast_values.std()
            metrics['forecast_min'] = forecast_values.min()
            metrics['forecast_max'] = forecast_values.max()
            
    return metrics

def print_forecast_summary(metrics: Dict, current_value: Optional[float] = None):
    """Print a formatted forecast summary"""
    print_subsection_header("🎯 FORECAST SUMMARY")
    
    if 'selected_model' in metrics:
        print(f"Best Model: {metrics['selected_model']}")
    
    if 'forecast_mean' in metrics:
        print(f"Forecast Average: {format_currency(metrics['forecast_mean'], 2)}")
        print(f"Forecast Range: {format_currency(metrics['forecast_min'], 2)} - {format_currency(metrics['forecast_max'], 2)}")
        print(f"Forecast Volatility: {format_currency(metrics['forecast_std'], 2)}")
        
        if current_value:
            change_pct = ((metrics['forecast_mean'] / current_value) - 1) * 100
            print(f"Expected Change: {format_percentage(change_pct)}")

def create_model_performance_plot(eval_df: pd.DataFrame, selected_model: str) -> plt.Figure:
    """Create a model performance comparison plot"""
    
    if eval_df.empty:
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get MASE scores
    mase_data = eval_df[eval_df['metric'] == 'MASE'].iloc[0] if 'MASE' in eval_df['metric'].values else None
    
    if mase_data is not None:
        model_cols = [col for col in eval_df.columns if col not in ['metric']]
        scores = [mase_data[col] for col in model_cols if col in mase_data.index]
        
        # Color the selected model differently
        colors = ['gold' if col == selected_model else 'lightblue' for col in model_cols]
        
        bars = ax.bar(range(len(model_cols)), scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Model Performance Comparison\n(MASE Score - Lower is Better)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('MASE Score')
        ax.set_xticks(range(len(model_cols)))
        ax.set_xticklabels(model_cols, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight the best model
        if selected_model in model_cols:
            best_idx = model_cols.index(selected_model)
            ax.annotate('SELECTED', xy=(best_idx, scores[best_idx]), 
                       xytext=(best_idx, scores[best_idx] + max(scores)*0.1),
                       ha='center', fontweight='bold', color='red',
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    return fig

def check_environment():
    """Check if the environment is set up correctly"""
    import os
    
    issues = []
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("⚠️ OPENAI_API_KEY environment variable not set")
    
    # Check required packages
    required_packages = ['pandas', 'matplotlib', 'seaborn', 'timecopilot', 'yfinance']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"⚠️ Package '{package}' not installed")
    
    if issues:
        print("🔧 ENVIRONMENT SETUP ISSUES:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Setup instructions:")
        print("   1. Set OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print("   2. Install packages: pip install -r requirements.txt")
        return False
    else:
        print("✅ Environment setup is complete!")
        return True

def generate_sample_questions(domain: str) -> List[str]:
    """Generate sample questions for different domains"""
    
    questions = {
        'stock': [
            "What are the biggest risks in this stock forecast?",
            "Should I buy, hold, or sell based on this forecast?",
            "How does recent performance compare to historical patterns?",
            "What external factors could make this forecast inaccurate?",
            "Which days are likely to have the highest prices?"
        ],
        'energy': [
            "What are the most critical hours for demand forecasting accuracy?",
            "How do seasonal patterns interact in this energy data?",
            "Which model is most suitable for grid operations and why?",
            "What are the main risks if we underestimate peak demand?",
            "How reliable is this forecast for electricity grid planning?"
        ],
        'ecommerce': [
            "What's the expected revenue and confidence interval?",
            "Which days are likely to have the highest sales?",
            "How should we adjust inventory based on this forecast?",
            "What are the main business risks if the forecast is wrong?",
            "What marketing strategies would you recommend?"
        ],
        'general': [
            "What patterns do you see in this time series?",
            "How confident are you in this forecast?",
            "What are the key drivers of these trends?",
            "What could cause this forecast to be inaccurate?",
            "How should we interpret these results for decision making?"
        ]
    }
    
    return questions.get(domain, questions['general'])

# Version info
__version__ = "1.0.0"
__author__ = "TimeCopilot Weekend Project"

if __name__ == "__main__":
    print("🛠️ TimeCopilot Weekend Project Utilities")
    print(f"Version: {__version__}")
    print("This module provides common utilities for the weekend project demos.")
    print("\nMain functions:")
    print("  • setup_plotting_style(): Configure consistent plot styling")
    print("  • print_section_header(): Print formatted headers")
    print("  • calculate_basic_stats(): Calculate time series statistics")
    print("  • create_forecast_comparison_plot(): Standard forecast visualization")
    print("  • check_environment(): Verify setup is complete")
