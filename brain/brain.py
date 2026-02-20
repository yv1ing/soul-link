from agents import Agent, Runner, function_tool
from config import settings
from memory import HybridMemory


class Brain:
    def __init__(self, session_id: str = "default"):
        self.memory = HybridMemory(session_id=session_id)
        self._soul_agent: Agent | None = None
        self._reflection_agent: Agent | None = None

    def think(self, text_input: str):
        result = Runner.run_sync(
            starting_agent=self._get_soul_agent(),
            input=text_input,
            session=self.memory,
        )
        return result.final_output

    def introspect(self):
        seed = self.memory.gather_reflection_seed()
        if not seed:
            print(f"[introspect] seed: None")
            return ""

        sections = []
        if "## Current user profile" in seed:
            sections.append("profile")
        if "## Summary of recent conversations" in seed:
            sections.append("episodes")
        if "## Fading memory" in seed:
            sections.append("fading")

        print(f"[introspect] seed: {', '.join(sections)}")
        # for line in seed.splitlines():
        #     print(f"  {line}")

        result = Runner.run_sync(
            starting_agent=self._get_reflection_agent(),
            input=seed,
        )

        self.memory.absorb_reflection(result.final_output)
        return result.final_output

    def close(self):
        self.memory.close()

    def _get_soul_agent(self) -> Agent:
        if self._soul_agent:
            return self._soul_agent

        self._soul_agent = Agent(
            name="Soul agent",
            model=settings.soul_model,
            instructions=settings.soul_prompt,
        )

        return self._soul_agent

    def _get_reflection_agent(self) -> Agent:
        if self._reflection_agent:
            return self._reflection_agent

        _memory = self.memory

        @function_tool
        def recall(query: str) -> str:
            """Search long-term memory by semantic similarity.

            Args:
                query: Natural language search query describing the topic or concept to look up.

            Returns:
                A newline-separated list of matching memory entries, each formatted as:
                "- [category | score: N.NN] abstract (uri: memory_uri)"
                Returns a "not found" message if no memories match.
            """

            results = _memory.persona.search(query)
            if settings.memory_decay_enabled:
                results = _memory.decay.apply_decay(results)
            if not results:
                print(f"[introspect] recall, query={query!r} -> empty")
                return "No relevant memories were found."

            output = "\n".join(f"- [{m.get('category') or 'uncategorized'} | score: {m['score']:.2f}] {m.get('abstract', '')} (uri: {m.get('uri', '')})" for m in results)

            print(f"[introspect] recall, query={query!r} -> {len(results)} hits")
            for line in output.splitlines():
                print(f"    {line}")
            return output

        @function_tool
        def recall_detail(uri: str) -> str:
            """Retrieve the full overview of a specific memory entry.

            Args:
                uri: The memory URI obtained from recall results (e.g. "viking://memories/...").

            Returns:
                The complete overview text of the memory. Returns an error message if
                the URI is invalid or inaccessible.
            """

            try:
                overview = _memory.persona.read_overview(uri) or "无详细内容。"
                print(f"[introspect] recall_detail, uri={uri} -> {len(overview)} chars")
                return overview
            except Exception as e:
                print(f"[introspect] recall_detail, uri={uri} -> FAIL: {e}")
                return "Failed to retrieve overview."

        @function_tool
        def reinforce(uri: str) -> str:
            """Reinforce a memory to prevent it from decaying. Resets the decay timer
            and increments the access count, increasing its long-term stability.

            Args:
                uri: The memory URI to reinforce (e.g. "viking://memories/...").

            Returns:
                A confirmation message indicating the memory has been reinforced.
            """

            _memory.decay.record_accesses([{"uri": uri}])

            print(f"[introspect] reinforce, uri={uri} -> ok")
            return "Memory reinforcement successful."

        @function_tool
        def forget(uri: str) -> str:
            """Permanently delete a memory from both the long-term store and decay metadata.
            This action is irreversible. Only use for memories that are outdated, incorrect,
            or no longer relevant.

            Args:
                uri: The memory URI to delete (e.g. "viking://memories/...").

            Returns:
                A confirmation message on success, or an error message if deletion failed.
            """

            try:
                _memory.persona.delete_memory(uri)
                _memory.decay.purge([uri])
                print(f"[introspect] forget, uri={uri} -> ok")
                return "The memory has been forgotten."
            except Exception as e:
                print(f"[introspect] forget, uri={uri} -> FAIL: {e}")
                return "The operation failed."

        self._reflection_agent = Agent(
            name="Reflection agent",
            model=settings.reflection_model,
            instructions=settings.reflection_prompt,
            tools=[recall, recall_detail, reinforce, forget],
        )

        return self._reflection_agent
