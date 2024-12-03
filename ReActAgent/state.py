import operator
from typing import Annotated, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
    """A state class for LangGraph agent execution.
    
    This class defines the structure of the state that is passed between nodes in the LangGraph workflow.
    It contains the input query, agent's reasoning outcome, and intermediate execution steps.

    Attributes:
        input: The input query string to be processed by the agent
        agent_outcome: The result of agent's reasoning, can be either:
            - AgentAction: Next action to be taken
            - AgentFinish: Final response when reasoning is complete
            - None: Initial state
        intermediate_steps: List of tuples containing agent actions and their results,
            accumulated using operator.add for combining multiple execution steps
    """
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
