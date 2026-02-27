"""Planning workflow implementation using LangGraph."""

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from agentic_ai_mcp.tools.registry import ToolRegistry
from agentic_ai_mcp.utils.retry import retry_with_backoff


class PlanningState(TypedDict):
    """State for the planning workflow."""

    task: str
    plan: list[str]
    current_step: int
    step_results: Annotated[list[str], operator.add]
    final_result: str


class PlanningWorkflow:
    """Planning workflow for complex multi-step tasks.

    Uses LangGraph StateGraph to:
    1. Plan: Break down the task into steps
    2. Execute: Run each step with tools
    3. Synthesize: Combine results into final response
    """

    def __init__(
        self,
        llm: Any,
        tools: list[StructuredTool],
        tool_registry: ToolRegistry,
        max_retries: int = 5,
        verbose: bool = False,
    ) -> None:
        """Initialize the planning workflow.

        Args:
            llm: LangChain chat model instance
            tools: List of LangChain tools
            tool_registry: Tool registry for signature formatting
            max_retries: Maximum retry attempts for API calls
            verbose: Enable verbose output
        """
        self.llm = llm
        self.tools = tools
        self.tool_registry = tool_registry
        self.max_retries = max_retries
        self.verbose = verbose
        self._workflow: Any = None

    def _build_workflow(self) -> Any:
        """Build the planning workflow graph.

        Returns:
            Compiled LangGraph workflow
        """
        llm = self.llm
        tools = self.tools
        verbose = self.verbose
        max_retries = self.max_retries
        tool_registry = self.tool_registry

        # Format tools with signatures
        tool_signatures = [tool_registry.format_tool_signature(t) for t in tools]

        # Planner node
        async def planner(state: PlanningState) -> dict[str, Any]:
            task = state["task"]

            plan_prompt = f"""Break down this task into clear, executable steps.
Each step should use the available tools efficiently.

Available tools:
{chr(10).join(f"  - {sig}" for sig in tool_signatures)}

Task: {task}

Respond with ONLY a numbered list of steps. Reference previous step results when needed.

Example for task "calculate 2+3, then greet Bob that many times":
1. Use add(a=2, b=3) to calculate the sum
2. Use greet(name="Bob", times=result from step 1) to greet Bob

Example for task "multiply 4 and 5, then add 10":
1. Use multiply(a=4, b=5) to get the product
2. Use add(a=result from step 1, b=10) to add 10"""

            async def invoke_planner() -> Any:
                return await llm.ainvoke([HumanMessage(content=plan_prompt)])

            response = await retry_with_backoff(invoke_planner, max_retries=max_retries, verbose=verbose)
            plan_text = str(response.content)

            # Parse steps
            steps = []
            for line in plan_text.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove numbering
                    step = line.lstrip("0123456789.").strip()
                    if step:
                        steps.append(step)

            if verbose:
                print("PLAN:")
                for i, step in enumerate(steps, 1):
                    print(f"  {i}. {step}")
                print()

            return {"plan": steps, "current_step": 0}

        # Executor node
        async def executor(state: PlanningState) -> dict[str, Any]:
            from langchain_core.messages import AIMessage, ToolMessage

            plan = state["plan"]
            current_step = state["current_step"]
            step_results = state.get("step_results", [])

            if current_step >= len(plan):
                return {"current_step": current_step}

            # Add delay between steps to avoid rate limiting (skip for first step)
            if current_step > 0:
                await asyncio.sleep(1.0)

            step = plan[current_step]

            if verbose:
                print(f"EXECUTING STEP {current_step + 1}/{len(plan)}: {step}")
                print("-" * 40)

            # Build context with previous results
            if step_results:
                context = "Previous results:\n" + "\n".join(step_results) + "\n\n"
                step_prompt = f"{context}Current task: {step}"
            else:
                step_prompt = step

            # Mini agent with retry logic for API overload
            step_agent = create_react_agent(llm, tools)

            async def invoke_step() -> Any:
                return await step_agent.ainvoke({"messages": [HumanMessage(content=step_prompt)]})

            result = await retry_with_backoff(invoke_step, max_retries=max_retries, verbose=verbose)

            # Extract result
            messages = result.get("messages", [])
            step_result = "No result"
            tool_call_count = 0

            for msg in messages:
                if isinstance(msg, AIMessage):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_call_count += 1
                            if verbose:
                                print(f"  TOOL CALL {tool_call_count}: {tc['name']}({tc['args']})")
                    elif msg.content:
                        step_result = str(msg.content)
                        if verbose:
                            print(f"  AI RESPONSE: {msg.content}")
                elif isinstance(msg, ToolMessage) and verbose:
                    print(f"    → {msg.content}")

            if verbose:
                print("-" * 40)
                print(f"STEP {current_step + 1} COMPLETE\n")

            return {
                "step_results": [f"Step {current_step + 1}: {step_result}"],
                "current_step": current_step + 1,
            }

        # Conditional check
        def should_continue(state: PlanningState) -> str:
            if state["current_step"] < len(state["plan"]):
                return "executor"
            return "synthesizer"

        # Synthesizer node
        async def synthesizer(state: PlanningState) -> dict[str, Any]:
            task = state["task"]
            step_results = state["step_results"]

            if verbose:
                print("=" * 50)
                print("SYNTHESIZING RESULTS")
                print("=" * 50)
                print(f"Original task: {task}")
                print()
                print("Step results collected:")
                for result in step_results:
                    print(f"  • {result}")
                print()

            synth_prompt = f"""You completed a multi-step task. Synthesize the results into a final response.

Original task: {task}

Step results:
{chr(10).join(step_results)}

Provide a clear, concise final response that addresses the original task."""

            if verbose:
                print("Sending synthesis prompt...")

            async def invoke_synthesizer() -> Any:
                return await llm.ainvoke([HumanMessage(content=synth_prompt)])

            response = await retry_with_backoff(invoke_synthesizer, max_retries=max_retries, verbose=verbose)

            if verbose:
                print("Synthesis complete.")
                print("=" * 50)

            return {"final_result": str(response.content)}

        # Build workflow graph
        workflow = StateGraph(PlanningState)
        workflow.add_node("planner", planner)
        workflow.add_node("executor", executor)
        workflow.add_node("synthesizer", synthesizer)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_conditional_edges("executor", should_continue)
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def get_workflow(self) -> Any:
        """Get or create the workflow.

        Returns:
            Compiled LangGraph workflow
        """
        if self._workflow is None:
            self._workflow = self._build_workflow()
        return self._workflow

    async def run(self, task: str) -> str:
        """Run the planning workflow.

        Args:
            task: Task description to execute

        Returns:
            Final synthesized result
        """
        if self.verbose:
            print(f"\n{'=' * 50}")
            print("PLANNING MODE")
            print(f"TASK: {task}")
            print(f"{'=' * 50}\n")

        workflow = self.get_workflow()

        initial_state: PlanningState = {
            "task": task,
            "plan": [],
            "current_step": 0,
            "step_results": [],
            "final_result": "",
        }

        result = await workflow.ainvoke(initial_state)

        if self.verbose:
            print(f"\n{'=' * 50}")
            print(f"FINAL RESULT: {result['final_result']}")
            print(f"{'=' * 50}\n")

        return str(result["final_result"])
