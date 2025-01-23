from gpt_researcher import GPTResearcher
from colorama import Fore, Style
from .utils.views import print_agent_output


class ResearchAgent:
    def __init__(self, websocket=None, stream_output=None, tone=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone

    async def research(self, query: str, research_report: str = "research_report",
                       parent_query: str = "", verbose=True, source="web", tone=None, headers=None):
        # Initialize the researcher
        researcher = GPTResearcher(query=query, report_type=research_report, parent_query=parent_query,
                                   verbose=verbose, report_source=source, tone=tone, websocket=self.websocket, headers=self.headers)
        # Conduct research on the given query
        await researcher.conduct_research()
        # Write the report
        report = await researcher.write_report()

        return report

    async def run_subtopic_research(self, parent_query: str, subtopic: str, verbose: bool = True, source="web", headers=None):
        try:
            report = await self.research(parent_query=parent_query, query=subtopic,
                                         research_report="subtopic_report", verbose=verbose, source=source, tone=self.tone, headers=None)
        except Exception as e:
            print(f"{Fore.RED}Error in researching topic {subtopic}: {e}{Style.RESET_ALL}")
            report = None
        return {subtopic: report}

    async def run_initial_research(self, research_state: dict):
        task = research_state.get("task")
        query = task.get("query")
        source = task.get("source", "web")

        if self.websocket and self.stream_output:
            await self.stream_output("logs", "initial_research", f"Running initial research on the following query: {query}", self.websocket)
        else:
            print_agent_output(f"Running initial research on the following query: {query}", agent="RESEARCHER")
        return {"task": task, "initial_research": await self.research(query=query, verbose=task.get("verbose"),
                                                                      source=source, tone=self.tone, headers=self.headers)}

    async def run_depth_research(self, draft_state: dict):
        """Research to find values and sources for simulation parameters."""
        if not draft_state:
            raise ValueError("draft_state cannot be None")
            
        task = draft_state.get("task")
        if not task:
            raise ValueError("task cannot be None in draft_state")
            
        parameter = draft_state.get("parameter")
        if not parameter:
            raise ValueError(f"parameter cannot be None or empty in draft_state. Full draft_state: {draft_state}")
            
        parent_query = task.get("query")
        if not parent_query:
            raise ValueError("query cannot be None in task")
            
        source = task.get("source", "web")
        verbose = task.get("verbose")
        
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "parameter_research", 
                f"Searching for value and source for parameter: {parameter}", self.websocket)
        else:
            print_agent_output(f"Searching for value and source for parameter: {parameter}", agent="RESEARCHER")
            
        # Construct a focused query to find the parameter value
        search_query = f"scientific research data value for {parameter} in population health studies"
        research_result = await self.research(
            query=search_query,
            research_report="parameter_value",
            verbose=verbose,
            source=source,
            headers=self.headers
        )
        
        # Extract parameter value and source from research
        parameter_data = {
            "name": parameter,
            "value": None,  # Will be extracted from research
            "unit": None,   # Will be extracted from research
            "source": None, # Will be extracted from research
            "confidence": "medium"  # Default confidence level
        }
        
        # Process research result to extract parameter info
        try:
            # Call model to extract structured parameter data from research
            prompt = [
                {
                    "role": "system",
                    "content": "You are a data extraction expert. Extract the most relevant parameter value and source from research findings."
                },
                {
                    "role": "user",
                    "content": f"""Research findings: {research_result}
                    Parameter to extract: {parameter}
                    
                    Extract and return a JSON with:
                    1. The numerical value found (if any)
                    2. The unit of measurement
                    3. The source URL or citation
                    4. Confidence level (high/medium/low) based on source quality
                    
                    Return format:
                    {{
                        "value": number,
                        "unit": "string",
                        "source": "string",
                        "confidence": "string"
                    }}"""
                }
            ]
            
            extracted_data = await call_model(
                prompt=prompt,
                model=task.get("model"),
                response_format="json"
            )
            
            parameter_data.update(extracted_data)
            
        except Exception as e:
            print(f"{Fore.RED}Error extracting parameter data: {e}{Style.RESET_ALL}")
            
        return {"parameter_data": parameter_data}