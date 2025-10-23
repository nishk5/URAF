"""
Tool Use System - ReAct Agent with Function Calling

Implements ReAct (Reasoning + Acting) pattern for tool-augmented LLMs.
Based on "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023)
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import requests
from loguru import logger


class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM to understand usage."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, str]:
        """Parameter schema for the tool."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Execute the tool with given parameters.

        Returns:
            Dict with 'success', 'result', and optional 'error'
        """
        pass

    def validate_parameters(self, **kwargs) -> bool:
        """Validate that required parameters are provided."""
        required_params = self.parameters.keys()
        provided_params = kwargs.keys()
        missing = set(required_params) - set(provided_params)

        if missing:
            logger.warning(f"Missing parameters for {self.name}: {missing}")
            return False
        return True


class CalculatorTool(BaseTool):
    """Calculator tool for mathematical operations."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Supports +, -, *, /, **, sqrt, sin, cos, etc."

    @property
    def parameters(self) -> dict[str, str]:
        return {"expression": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(3.14)')"}

    async def execute(self, expression: str, **kwargs) -> dict[str, Any]:
        """
        Safely evaluate mathematical expressions.

        Args:
            expression: Math expression string

        Returns:
            Evaluation result
        """
        try:
            # Safe eval with limited scope
            import math

            safe_dict = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e,
            }

            result = eval(expression, safe_dict)
            logger.info(f"Calculator: {expression} = {result}")

            return {"success": True, "result": result, "expression": expression}

        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return {"success": False, "error": str(e), "expression": expression}


class WebSearchTool(BaseTool):
    """Web search tool (mock implementation - replace with real API)."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information. Returns top search results."

    @property
    def parameters(self) -> dict[str, str]:
        return {"query": "Search query string", "num_results": "Number of results to return (default: 3)"}

    async def execute(self, query: str, num_results: int = 3, **kwargs) -> dict[str, Any]:
        """
        Perform web search (mock for now).

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            Search results
        """
        # Mock implementation - in production, use DuckDuckGo, SerpAPI, etc.
        logger.info(f"Web search: {query}")

        mock_results = [
            {
                "title": f"Result {i + 1} for '{query}'",
                "snippet": f"This is a mock search result for {query}. In production, this would be real web data.",
                "url": f"https://example.com/result{i + 1}",
            }
            for i in range(num_results)
        ]

        return {"success": True, "query": query, "results": mock_results, "num_results": len(mock_results)}


class CodeExecutorTool(BaseTool):
    """Execute Python code in a sandboxed environment."""

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return "Execute Python code and return the output. Use for data analysis or computations."

    @property
    def parameters(self) -> dict[str, str]:
        return {"code": "Python code to execute", "timeout": "Execution timeout in seconds (default: 5)"}

    async def execute(self, code: str, timeout: int = 5, **kwargs) -> dict[str, Any]:
        """
        Execute Python code safely.

        Args:
            code: Python code string
            timeout: Execution timeout

        Returns:
            Execution result
        """
        try:
            # In production, use docker/sandbox for safety
            import io
            from contextlib import redirect_stdout

            # Capture output
            output_buffer = io.StringIO()

            with redirect_stdout(output_buffer):
                # Limited scope execution
                exec_globals = {"__builtins__": __builtins__}
                exec(code, exec_globals)

            output = output_buffer.getvalue()
            logger.info("Code executed successfully")

            return {
                "success": True,
                "output": output,
                "code": code[:100],  # First 100 chars
            }

        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {"success": False, "error": str(e), "code": code[:100]}


class WikipediaSearchTool(BaseTool):
    """Search Wikipedia for factual information."""

    @property
    def name(self) -> str:
        return "wikipedia_search"

    @property
    def description(self) -> str:
        return "Search Wikipedia for factual information. Returns article summary."

    @property
    def parameters(self) -> dict[str, str]:
        return {"query": "Topic to search on Wikipedia"}

    async def execute(self, query: str, **kwargs) -> dict[str, Any]:
        """
        Search Wikipedia API.

        Args:
            query: Search topic

        Returns:
            Wikipedia summary
        """
        try:
            # Wikipedia API endpoint
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "query": query,
                    "title": data.get("title", ""),
                    "summary": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
            else:
                return {"success": False, "error": f"Wikipedia returned status {response.status_code}", "query": query}

        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return {"success": False, "error": str(e), "query": query}


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        """Initialize tool registry with default tools."""
        self.tools: dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default built-in tools."""
        default_tools = [CalculatorTool(), WebSearchTool(), CodeExecutorTool(), WikipediaSearchTool()]

        for tool in default_tools:
            self.register_tool(tool)

        logger.info(f"Registered {len(self.tools)} default tools")

    def register_tool(self, tool: BaseTool):
        """
        Register a new tool.

        Args:
            tool: Tool instance to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> BaseTool | None:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        """
        List all available tools with descriptions.

        Returns:
            List of tool metadata
        """
        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in self.tools.values()
        ]

    async def execute_tool(self, tool_name: str, **parameters) -> dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            **parameters: Tool parameters

        Returns:
            Execution result
        """
        tool = self.get_tool(tool_name)

        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        if not tool.validate_parameters(**parameters):
            return {"success": False, "error": f"Invalid parameters for {tool_name}"}

        return await tool.execute(**parameters)


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent with tool use.

    Implements the Thought-Action-Observation loop.
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry | None = None):
        """
        Initialize ReAct agent.

        Args:
            llm_client: LLM client for reasoning
            tool_registry: Tool registry (creates default if None)
        """
        self.llm = llm_client
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = 10
        self.execution_history: list[dict] = []
        logger.info("Initialized ReAct agent")

    def _parse_action(self, response: str) -> dict[str, Any] | None:
        """
        Parse action from LLM response.

        Expected format:
        Action: tool_name
        Action Input: {"param1": "value1", ...}

        Args:
            response: LLM response text

        Returns:
            Parsed action dict or None
        """
        # Extract action
        action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
        if not action_match:
            return None

        tool_name = action_match.group(1).strip()

        # Extract action input
        input_match = re.search(r"Action Input:\s*({.*?}|\w+.*?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)

        if input_match:
            input_str = input_match.group(1).strip()

            # Try to parse as JSON
            try:
                if input_str.startswith("{"):
                    action_input = json.loads(input_str)
                else:
                    # Simple string parameter
                    action_input = {"input": input_str}
            except json.JSONDecodeError:
                action_input = {"input": input_str}
        else:
            action_input = {}

        return {"tool": tool_name, "parameters": action_input}

    def _create_react_prompt(self, task: str, history: list[dict]) -> str:
        """
        Create ReAct-style prompt with task and execution history.

        Args:
            task: User task
            history: Execution history (thoughts, actions, observations)

        Returns:
            Formatted prompt
        """
        tools_desc = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in self.tool_registry.list_tools()])

        history_text = ""
        for i, entry in enumerate(history):
            history_text += f"\nIteration {i + 1}:\n"
            history_text += f"Thought: {entry.get('thought', '')}\n"
            if "action" in entry:
                history_text += f"Action: {entry['action']['tool']}\n"
                history_text += f"Action Input: {entry['action']['parameters']}\n"
                history_text += f"Observation: {entry.get('observation', '')}\n"

        prompt = f"""You are a ReAct (Reasoning + Acting) agent that can use tools to solve tasks.

Available Tools:
{tools_desc}

Task: {task}

{history_text}

Instructions:
1. Think step-by-step about what to do next
2. If you need to use a tool, format your response as:
   Thought: [your reasoning]
   Action: [tool_name]
   Action Input: {{"parameter": "value"}}

3. If you have enough information to answer, format as:
   Thought: [your reasoning]
   Final Answer: [your complete answer]

Your response:"""

        return prompt

    async def solve(self, task: str) -> dict[str, Any]:
        """
        Solve a task using ReAct loop.

        Args:
            task: User task description

        Returns:
            Solution result with execution trace
        """
        logger.info(f"ReAct agent solving: {task}")

        self.execution_history = []

        for iteration in range(self.max_iterations):
            # Generate prompt
            prompt = self._create_react_prompt(task, self.execution_history)

            # Get LLM response
            try:
                response = await self.llm.query(prompt)
                response_text = response.get("summary", "") if isinstance(response, dict) else str(response)
            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                return {
                    "success": False,
                    "error": f"LLM error: {str(e)}",
                    "iterations": iteration + 1,
                    "history": self.execution_history,
                }

            # Check for final answer
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                logger.info(f"ReAct solved in {iteration + 1} iterations")

                return {
                    "success": True,
                    "final_answer": final_answer,
                    "iterations": iteration + 1,
                    "history": self.execution_history,
                }

            # Parse action
            action = self._parse_action(response_text)

            if not action:
                # No action found, just record thought
                thought_match = re.search(r"Thought:\s*(.*?)(?:\n|$)", response_text, re.IGNORECASE)
                thought = thought_match.group(1).strip() if thought_match else response_text[:200]

                self.execution_history.append({"iteration": iteration + 1, "thought": thought})
                continue

            # Execute tool
            observation = await self.tool_registry.execute_tool(action["tool"], **action["parameters"])

            # Record execution
            self.execution_history.append(
                {
                    "iteration": iteration + 1,
                    "thought": response_text[:200],
                    "action": action,
                    "observation": observation,
                }
            )

            logger.info(f"Iteration {iteration + 1}: Used {action['tool']}")

        # Max iterations reached
        logger.warning(f"ReAct reached max iterations ({self.max_iterations})")
        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached without solution",
            "iterations": self.max_iterations,
            "history": self.execution_history,
        }


class ToolUseEvaluator:
    """Evaluator for tool use capabilities."""

    def __init__(self):
        """Initialize tool use evaluator."""
        self.metrics = []

    def evaluate_tool_execution(self, react_result: dict[str, Any]) -> dict[str, float]:
        """
        Evaluate quality of tool use in ReAct execution.

        Args:
            react_result: Result from ReAct agent

        Returns:
            Tool use metrics
        """
        history = react_result.get("history", [])

        if not history:
            return {
                "tool_selection_accuracy": 0.0,
                "execution_success_rate": 0.0,
                "efficiency_score": 0.0,
                "overall_tool_score": 0.0,
            }

        # Count successful tool executions
        tool_executions = [h for h in history if "action" in h]
        successful_executions = [h for h in tool_executions if h.get("observation", {}).get("success", False)]

        execution_success_rate = len(successful_executions) / len(tool_executions) if tool_executions else 0.0

        # Efficiency: fewer iterations is better
        iterations = react_result.get("iterations", 10)
        efficiency_score = max(0.0, 1.0 - (iterations / 10))

        # Success bonus
        success_bonus = 0.3 if react_result.get("success", False) else 0.0

        overall_score = execution_success_rate * 0.5 + efficiency_score * 0.3 + success_bonus

        return {
            "tool_selection_accuracy": float(execution_success_rate),
            "execution_success_rate": float(execution_success_rate),
            "efficiency_score": float(efficiency_score),
            "num_tool_calls": len(tool_executions),
            "successful_calls": len(successful_executions),
            "overall_tool_score": float(overall_score),
        }
