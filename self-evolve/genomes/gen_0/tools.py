"""Genome-defined extra tools.

The candidate agent loads any function defined here whose name starts with
`tool_` and exposes it as a callable tool. The function's docstring becomes
the tool description. Parameters must be JSON-serializable (str/int/bool/list/dict).

Initially empty — the orchestrator's Planner can author new tools here.

Convention:
    def tool_my_thing(arg1: str, arg2: int = 0) -> str:
        '''One-line description shown to the agent.'''
        ...
        return "result string"
"""
