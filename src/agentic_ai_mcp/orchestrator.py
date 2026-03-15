"""AgenticAIOrchestrator - Coordinates multiple AgenticAIClient instances."""

import asyncio
from typing import Any, Literal

from langchain_core.messages import HumanMessage

from agentic_ai_mcp.client import AgenticAIClient
from agentic_ai_mcp.shared_state import SharedState


class AgenticAIOrchestrator:
    """Orchestrates multiple AgenticAIClient instances to work together on a task.

    Supports sequential and parallel execution flows, with an optional
    synthesize step that uses a plain LLM (no tools) to combine results.

    Example:
        researcher = AgenticAIClient(role="researcher", tool_filter=["search"])
        writer = AgenticAIClient(role="writer", tool_filter=["write_file"])

        orchestrator = AgenticAIOrchestrator(
            clients=[researcher, writer],
            flow_type="sequential",
            synthesize=True,
        )
        result = await orchestrator.run("Research and document the project")
    """

    def __init__(
        self,
        clients: list[AgenticAIClient],
        flow_type: Literal["sequential", "parallel"] = "sequential",
        synthesize: bool = False,
        shared_state: SharedState | None = None,
    ) -> None:
        """Initialize AgenticAIOrchestrator.

        Args:
            clients: List of AgenticAIClient instances to orchestrate.
            flow_type: Execution flow - "sequential" or "parallel".
            synthesize: If True, a plain LLM call (no tools) combines
                all agent results into a final response.
            shared_state: Optional SharedState for agents to read/write.
                If not provided, a new empty SharedState is created.
        """
        if not clients:
            raise ValueError("At least one client is required")

        self._clients = clients
        self._flow_type = flow_type
        self._do_synthesize = synthesize
        self._shared_state = shared_state or SharedState()

        # attach shared state to all clients
        for client in self._clients:
            client.shared_state = self._shared_state._data

    @property
    def clients(self) -> list[AgenticAIClient]:
        """List of orchestrated clients."""
        return self._clients.copy()

    @property
    def shared_state(self) -> SharedState:
        """The shared state instance."""
        return self._shared_state

    def _build_prompt(self, client: AgenticAIClient, prompt: str, context: str = "") -> str:
        """Build a prompt for a client, prepending role and context.

        Args:
            client: The client to build the prompt for.
            prompt: The original user prompt.
            context: Additional context from previous agents.

        Returns:
            The constructed prompt string.
        """
        parts: list[str] = []

        if client.role:
            parts.append(f"You are a {client.role}.")

        parts.append(
            "Only perform the parts of the task that can be accomplished with your available tools."
            " Do not attempt parts that require tools you do not have."
        )

        if context:
            parts.append(f"Previous agent results:\n{context}")

        parts.append(prompt)
        return "\n\n".join(parts)

    async def _run_sequential(
        self, prompt: str, use_planning: bool = False
    ) -> list[dict[str, Any]]:
        """Run clients sequentially, passing output forward as context.

        Args:
            prompt: The user prompt.
            use_planning: If True, use run_with_planning on each client.

        Returns:
            List of result dicts with role and output for each client.
        """
        results: list[dict[str, Any]] = []
        context = ""

        for client in self._clients:
            full_prompt = self._build_prompt(client, prompt, context)

            if use_planning:
                output = await client.run_with_planning(full_prompt)
            else:
                output = await client.run(full_prompt)

            results.append(
                {
                    "role": client.role or client.name,
                    "output": output,
                }
            )
            context = output

        return results

    async def _run_parallel(self, prompt: str, use_planning: bool = False) -> list[dict[str, Any]]:
        """Run all clients concurrently, collecting results.

        Args:
            prompt: The user prompt.
            use_planning: If True, use run_with_planning on each client.

        Returns:
            List of result dicts with role and output for each client.
        """

        async def _run_one(client: AgenticAIClient) -> dict[str, Any]:
            full_prompt = self._build_prompt(client, prompt)
            if use_planning:
                output = await client.run_with_planning(full_prompt)
            else:
                output = await client.run(full_prompt)
            return {
                "role": client.role or client.name,
                "output": output,
            }

        results = await asyncio.gather(*[_run_one(c) for c in self._clients])
        return list(results)

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """Format agent results into a combined string.

        Args:
            results: List of result dicts from agents.

        Returns:
            Formatted string with labeled results.
        """
        parts = []
        for r in results:
            parts.append(f"[{r['role']}]\n{r['output']}")
        return "\n\n".join(parts)

    async def _synthesize(self, prompt: str, combined: str) -> str:
        """Synthesize agent results using a plain LLM call (no tools).

        Uses the LLM from the first client to combine all agent results
        into a final response.

        Args:
            prompt: The original user prompt.
            combined: Formatted results from all agents.

        Returns:
            Synthesized final response.
        """
        synth_prompt = (
            f"You are a synthesizer. The following agents have worked on a task. "
            f"Combine their results into a clear, concise final response.\n\n"
            f"Original task: {prompt}\n\n"
            f"Agent results:\n{combined}\n\n"
            f"Provide a final synthesized response that addresses the original task."
        )

        llm = self._clients[0]._get_llm()
        response = await llm.ainvoke([HumanMessage(content=synth_prompt)])
        return str(response.content)

    async def _execute(self, prompt: str, use_planning: bool = False) -> str:
        """Core execution logic for both run modes.

        Args:
            prompt: The user prompt.
            use_planning: If True, use run_with_planning on each client.

        Returns:
            Final result string.
        """
        # Run main clients
        if self._flow_type == "sequential":
            results = await self._run_sequential(prompt, use_planning)
        else:
            results = await self._run_parallel(prompt, use_planning)

        # Store results in shared state
        self._shared_state.set("agent_results", results)

        # Format results
        combined = self._format_results(results)

        # If synthesize is enabled, run a plain LLM call to combine results
        if self._synthesize_enabled:
            return await self._synthesize(prompt, combined)

        return combined

    @property
    def _synthesize_enabled(self) -> bool:
        """Check if synthesis is enabled."""
        return self._do_synthesize

    async def run(self, prompt: str) -> str:
        """Run orchestration using simple ReAct on each client.

        Args:
            prompt: Task for the agents.

        Returns:
            Final result (synthesized if enabled,
            otherwise combined results from all agents).
        """
        return await self._execute(prompt, use_planning=False)

    async def run_with_planning(self, prompt: str) -> str:
        """Run orchestration using planning workflow on each client.

        Args:
            prompt: Task for the agents.

        Returns:
            Final result (synthesized if enabled,
            otherwise combined results from all agents).
        """
        return await self._execute(prompt, use_planning=True)

    def run_sync(self, prompt: str) -> str:
        """Synchronous version of run().

        Args:
            prompt: Task for the agents.

        Returns:
            Final result string.
        """
        return asyncio.run(self.run(prompt))

    def run_with_planning_sync(self, prompt: str) -> str:
        """Synchronous version of run_with_planning().

        Args:
            prompt: Task for the agents.

        Returns:
            Final result string.
        """
        return asyncio.run(self.run_with_planning(prompt))
