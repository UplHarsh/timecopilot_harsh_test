#!/bin/bash

# TimeCopilot Weekend Project Setup Script
# ========================================

echo "🚀 Setting up TimeCopilot Weekend Project..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "   Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is required but not installed."
    echo "   Please install pip and try again."
    exit 1
fi

# Create virtual environment (optional but recommended)
if [ "$1" == "--venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv timecopilot_weekend_env
    source timecopilot_weekend_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install required packages
echo "📦 Installing required packages..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt --quiet
else
    pip install -r requirements.txt --quiet
fi

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully"
else
    echo "❌ Error installing packages. Please check your internet connection and try again."
    exit 1
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ Warning: OPENAI_API_KEY environment variable not set"
    echo "   You'll need to set this to run the AI-powered features:"
    echo "   export OPENAI_API_KEY='your-key-here'"
    echo
else
    echo "✅ OpenAI API key detected"
fi

# Create output directory for visualizations
mkdir -p outputs
echo "✅ Output directory created"

echo
echo "🎉 Setup complete! You can now run:"
echo "   • python bitcoin_analysis.py      - Bitcoin price forecasting"
echo "   • python energy_demand.py         - Energy demand forecasting" 
echo "   • python ecommerce_sales.py       - E-commerce sales forecasting"
echo "   • jupyter notebook interactive_playground.ipynb - Interactive exploration"
echo
echo "📚 Don't forget to check README.md for detailed instructions!"
echo "⚡ Happy forecasting with TimeCopilot!"
