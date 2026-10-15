"""LangChain v1 wiring: the model and the `create_agent` agent.

LangChain v1 note
-----------------

`bind_tools` only told the model which tools exist; we still had to run the
loop ourselves. `create_agent` builds the whole loop -- model call, tool call,
model call again. The LCEL RAG chain is gone because retrieval is now the
`search_handbook` tool.
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .config import LLM_MODEL, TEMPERATURE
from .prompts import HR_SYSTEM_PROMPT
from .rag import build_handbook_tool
from .tools import HR_TOOLS

#Specific to debug middleware
from langchain.agents.middleware import AgentMiddleware
from pprint import pprint


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)


def build_agent(retriever):
    """Build the HR agent: model + tools + system prompt."""
    # The handbook retriever becomes one more tool alongside the 7 HR tools.
    # That single list is what makes the routing work -- the model picks one.
    handbook_tool = build_handbook_tool(retriever)

    return create_agent(
        model=build_llm(),
        tools=[handbook_tool, *HR_TOOLS],
        system_prompt=HR_SYSTEM_PROMPT,
        middleware=[LLMDebugMiddleware()],
    )

#Specific to debug middleware

class LLMDebugMiddleware(AgentMiddleware):

    def wrap_model_call(self, request, handler):

        # BREAKPOINT 1: Before every LLM call
        print("\n===== REQUEST SENT TO LLM =====")
        pprint(request.messages, sort_dicts=False)
        breakpoint()

        # The actual LLM call happens here
        response = handler(request)

        # BREAKPOINT 2: After every LLM call
        print("\n===== RAW LLM RESPONSE =====")
        pprint(response, sort_dicts=False)
        breakpoint()

        # Return the response so the agent can continue
        return response