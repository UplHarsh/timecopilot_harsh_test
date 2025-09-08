from .explainer import ForecastExplainer, ForecastExplanation
from .forecasting_example import TimeCopilotForecastingPipeline

# Conditional imports for modules with heavy dependencies
try:
    from .agent import AsyncTimeCopilot, TimeCopilot
    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False

try:
    from .forecaster import TimeCopilotForecaster
    _FORECASTER_AVAILABLE = True
except ImportError:
    _FORECASTER_AVAILABLE = False

__all__ = [
    "ForecastExplainer",
    "ForecastExplanation", 
    "TimeCopilotForecastingPipeline"
]

if _AGENT_AVAILABLE:
    __all__.extend(["AsyncTimeCopilot", "TimeCopilot"])

if _FORECASTER_AVAILABLE:
    __all__.extend(["TimeCopilotForecaster"])
