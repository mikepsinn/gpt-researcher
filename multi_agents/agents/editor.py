from datetime import datetime
import asyncio
from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END

from .utils.views import print_agent_output
from .utils.llms import call_model
from ..memory.research import ParameterState
from . import ResearchAgent, ReviewerAgent, ReviserAgent


class EditorAgent:
    """Agent responsible for editing and managing code."""

    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def plan_research(self, research_state: Dict[str, any]) -> Dict[str, any]:
        """
        Plan the simulation metrics and calculations.

        :param research_state: Dictionary containing research state information
        :return: Dictionary with title, date, and planned metrics
        """
        initial_research = research_state.get("initial_research")
        task = research_state.get("task")
        include_human_feedback = task.get("include_human_feedback")
        human_feedback = research_state.get("human_feedback")
        max_sections = task.get("max_sections")

        prompt = self._create_planning_prompt(
            initial_research, include_human_feedback, human_feedback, max_sections)

        print_agent_output(
            "Planning metrics and calculations for the simulation model...", agent="EDITOR")
        plan = await call_model(
            prompt=prompt,
            model=task.get("model"),
            response_format="json",
        )

        # Initialize the simulation state
        return {
            "title": plan.get("title"),
            "date": plan.get("date"),
            "outcome_metrics": plan.get("outcome_metrics", []),
            "equations": {},  # Will be populated based on metrics
            "parameters": {},  # Will be populated by parameter research
            "calculations": {},  # Will be populated after parameters are found
        }

    async def run_parallel_research(self, research_state: Dict[str, any]) -> Dict[str, List[str]]:
        """
        Execute parallel parameter research tasks.

        :param research_state: Dictionary containing research state information
        :return: Dictionary with parameter values and sources
        """
        if not research_state:
            raise ValueError("research_state cannot be None")

        # Get the metrics and their parameters from the planning stage
        metrics = research_state.get("outcome_metrics", [])
        if not metrics:
            raise ValueError("No metrics found in research_state. This indicates a problem with the planning stage.")

        # Extract all unique parameters needed across all metrics
        all_parameters = set()
        for i, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                raise ValueError(f"Metric at index {i} is not a dictionary: {metric}")
            if "parameters" not in metric:
                raise ValueError(f"Metric at index {i} has no parameters field: {metric}")
            parameters = metric.get("parameters", [])
            if not parameters:
                raise ValueError(f"Metric at index {i} has empty parameters: {metric}")
            all_parameters.update(parameters)

        if not all_parameters:
            raise ValueError("No parameters found across all metrics. This indicates a problem with the metrics definition.")

        print_agent_output(f"Found parameters to research: {list(all_parameters)}", agent="EDITOR")
        self._log_parallel_research(list(all_parameters))

        # Initialize agents and workflow
        agents = self._initialize_agents()
        workflow = self._create_workflow()
        chain = workflow.compile()

        # Create research tasks for each parameter
        final_parameter_tasks = []
        for parameter in all_parameters:
            if not parameter:
                raise ValueError("Found empty parameter in all_parameters")
            task_input = self._create_parameter_task_input(research_state, parameter)
            final_parameter_tasks.append(chain.ainvoke(task_input))

        # Gather all parameter research results
        parameter_results = await asyncio.gather(*final_parameter_tasks)
        
        # Combine parameter data into a single dictionary
        parameters = {}
        for result in parameter_results:
            if not result:
                continue
            if "parameter_data" not in result:
                print_agent_output(f"Warning: Missing parameter_data in result: {result}", agent="EDITOR")
                continue
            param_data = result["parameter_data"]
            if not param_data.get("name"):
                print_agent_output(f"Warning: Missing parameter name in data: {param_data}", agent="EDITOR")
                continue
            parameters[param_data["name"]] = {
                "value": param_data.get("value"),
                "unit": param_data.get("unit"),
                "source": param_data.get("source"),
                "confidence": param_data.get("confidence", "low")
            }

        if not parameters:
            raise ValueError("No parameter data was successfully collected. This indicates a problem with the research phase.")

        return {"parameters": parameters}

    def _create_planning_prompt(self, initial_research: str, include_human_feedback: bool,
                                human_feedback: Optional[str], max_sections: int) -> List[Dict[str, str]]:
        """Create the prompt for research planning."""
        return [
            {
                "role": "system",
                "content": "You are a quantitative impact modeling expert. Your goal is to oversee the development "
                           "of a mathematical model to estimate population-level health and economic impacts. "
                           "Your main task is to identify key outcome metrics and plan how to calculate them.\n ",
            },
            {
                "role": "user",
                "content": self._format_planning_instructions(initial_research, include_human_feedback,
                                                              human_feedback, max_sections),
            },
        ]

    def _format_planning_instructions(self, initial_research: str, include_human_feedback: bool,
                                      human_feedback: Optional[str], max_sections: int) -> str:
        """Format the instructions for simulation planning."""
        today = datetime.now().strftime('%d/%m/%Y')
        feedback_instruction = (
            f"Human feedback: {human_feedback}. You must plan the metrics based on the human feedback."
            if include_human_feedback and human_feedback and human_feedback != 'no'
            else ''
        )

        return f"""Today's date is {today}
                   Initial research summary: '{initial_research}'
                   {feedback_instruction}
                   \nYour task is to identify the key outcome metrics for estimating population-level health and economic impacts.
                   You must generate a maximum of {max_sections} high-level outcome metrics.
                   For each metric, you must specify:
                   1. The metric name and description
                   2. The units of measurement
                   3. The equation or formula to calculate it
                   4. The required input parameters
                   
                   You must return nothing but a JSON with the following structure:
                   {{
                     "title": "string title describing the simulation model",
                     "date": "today's date",
                     "outcome_metrics": [
                       {{
                         "name": "metric name",
                         "description": "metric description",
                         "unit": "unit of measurement",
                         "equation": "mathematical formula",
                         "parameters": ["list of required parameter names"]
                       }}
                     ]
                   }}"""

    def _initialize_agents(self) -> Dict[str, any]:
        """Initialize the research, reviewer, and reviser skills."""
        return {
            "research": ResearchAgent(self.websocket, self.stream_output, self.headers),
            "reviewer": ReviewerAgent(self.websocket, self.stream_output, self.headers),
            "reviser": ReviserAgent(self.websocket, self.stream_output, self.headers),
        }

    def _create_workflow(self) -> StateGraph:
        """Create the workflow for parameter research."""
        agents = self._initialize_agents()
        workflow = StateGraph(ParameterState)

        # We only need the researcher node since we're just finding parameter values
        workflow.add_node("researcher", agents["research"].run_depth_research)
        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", END)

        return workflow

    def _log_parallel_research(self, queries: List[str]) -> None:
        """Log the start of parallel research tasks."""
        if self.websocket and self.stream_output:
            asyncio.create_task(self.stream_output(
                "logs",
                "parallel_research",
                f"Running parallel research for the following queries: {queries}",
                self.websocket,
            ))
        else:
            print_agent_output(
                f"Running the following research tasks in parallel: {queries}...",
                agent="EDITOR",
            )

    def _create_parameter_task_input(self, research_state: Dict[str, any], parameter: str) -> Dict[str, any]:
        """Create the input for a parameter research task."""
        if not research_state:
            raise ValueError("research_state cannot be None")
        
        task = research_state.get("task")
        if not task:
            raise ValueError("task cannot be None in research_state")
            
        if not parameter:
            raise ValueError("parameter cannot be None or empty")
            
        # Create a proper ParameterState
        return {
            "task": task,
            "parameter": parameter,
            "parameter_data": {}  # Will be populated by researcher
        }
