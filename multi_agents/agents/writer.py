from datetime import datetime
import json5 as json
from .utils.views import print_agent_output
from .utils.llms import call_model

sample_json = """
{
  "table_of_contents": A table of contents in markdown syntax (using '-') based on the metrics and calculations,
  "introduction": An introduction explaining the simulation model's purpose and approach,
  "metrics_documentation": A detailed documentation of each metric including:
    - Description and rationale
    - Equation and parameters
    - Parameter values and sources
    - Calculation results and assumptions,
  "conclusion": A summary of the key findings and potential impact,
  "sources": A list of all parameter sources in APA format
}
"""


class WriterAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers

    def get_headers(self, research_state: dict):
        return {
            "title": research_state.get("title"),
            "date": "Date",
            "introduction": "Introduction",
            "table_of_contents": "Table of Contents",
            "metrics": "Impact Metrics and Calculations",
            "conclusion": "Conclusion",
            "references": "Parameter Sources",
        }

    async def write_sections(self, research_state: dict):
        """Write the simulation model documentation."""
        metrics = research_state.get("outcome_metrics", [])
        parameters = research_state.get("parameters", {})
        calculations = research_state.get("calculations", {})
        task = research_state.get("task")
        
        prompt = [
            {
                "role": "system",
                "content": "You are a technical writer specializing in quantitative impact modeling. "
                "Your goal is to clearly document simulation models, their calculations, and results.",
            },
            {
                "role": "user",
                "content": f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}\n
                Simulation metrics: {json.dumps(metrics, indent=2)}
                Parameter values: {json.dumps(parameters, indent=2)}
                Calculation results: {json.dumps(calculations, indent=2)}
                
                Write a clear technical document that:
                1. Introduces the simulation model's purpose
                2. Documents each metric's calculation approach
                3. Lists parameter values and their sources
                4. Presents the calculated results
                5. Summarizes the key findings
                
                The document should be suitable for government health agencies.
                Include hyperlinks to parameter sources.
                
                Return a JSON matching this format:
                {sample_json}
                """,
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
        )
        return response

    async def revise_headers(self, task: dict, headers: dict):
        prompt = [
            {
                "role": "system",
                "content": """You are a research writer. 
Your sole purpose is to revise the headers data based on the given guidelines.""",
            },
            {
                "role": "user",
                "content": f"""Your task is to revise the given headers JSON based on the guidelines given.
You are to follow the guidelines but the values should be in simple strings, ignoring all markdown syntax.
You must return nothing but a JSON in the same format as given in headers data.
Guidelines: {task.get("guidelines")}\n
Headers Data: {headers}\n
""",
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
        )
        return {"headers": response}

    async def run(self, research_state: dict):
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "writing_report",
                f"Writing final research report based on research data...",
                self.websocket,
            )
        else:
            print_agent_output(
                f"Writing final research report based on research data...",
                agent="WRITER",
            )

        research_layout_content = await self.write_sections(research_state)

        if research_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                research_layout_content_str = json.dumps(
                    research_layout_content, indent=2
                )
                await self.stream_output(
                    "logs",
                    "research_layout_content",
                    research_layout_content_str,
                    self.websocket,
                )
            else:
                print_agent_output(research_layout_content, agent="WRITER")

        headers = self.get_headers(research_state)
        if research_state.get("task").get("follow_guidelines"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "rewriting_layout",
                    "Rewriting layout based on guidelines...",
                    self.websocket,
                )
            else:
                print_agent_output(
                    "Rewriting layout based on guidelines...", agent="WRITER"
                )
            headers = await self.revise_headers(
                task=research_state.get("task"), headers=headers
            )
            headers = headers.get("headers")

        return {**research_layout_content, "headers": headers}
