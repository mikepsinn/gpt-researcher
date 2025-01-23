from typing import TypedDict, List, Dict, Any, Annotated
import operator


class ParameterState(TypedDict):
    """State for parameter research tasks."""
    task: dict
    parameter: str  # The parameter being researched
    parameter_data: Dict[str, Any]  # The collected parameter data


class ResearchState(TypedDict):
    task: dict
    initial_research: str
    sections: List[str]
    research_data: List[dict]
    human_feedback: str
    # Simulation specific data
    outcome_metrics: List[Dict[str, Any]]  # List of metrics with descriptions and units
    equations: Dict[str, str]  # Metric name -> equation string
    parameters: Dict[str, Dict[str, Any]]  # Parameter name -> {value, source, description, unit}
    calculations: Dict[str, Dict[str, Any]]  # Metric name -> calculated results and assumptions
    # Report layout
    title: str
    headers: dict
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: List[str]
    report: str


