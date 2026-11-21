import re
import httpx
import os
from uuid import uuid4
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
import operator
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from dotenv import load_dotenv
_ = load_dotenv()

MODEL = "openai:gpt-6-luna"


def reduce_message(
        left: list[AnyMessage],
        right: list[AnyMessage]
) -> list[AnyMessage]:

    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())

    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_message]
    cancelled: bool


class Agent:

    def __init__(
       self,
       model: str,
       tools: list,
       checkpointer: InMemorySaver,
       system: str = ""
    ) -> None:
        
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_LLM)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_conditional_edges(
            "action",
            self.route_after_action,
            {True: END, False: "llm"}
        )
        graph.add_edge(START, "llm")
        self.graph = graph.compile(
            checkpointer=checkpointer
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)


    def call_LLM(
            self,
            state: AgentState
    ) -> dict:
        
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}


    def take_action(
            self,
            state: AgentState
    ) -> dict:

        tool_calls = state['messages'][-1].tool_calls
        print("Tool calls:", tool_calls)

        for t in tool_calls:
            print("ID:", t.get("id"))
            print("Name:", t.get("name"))

        while True:
            answer = interrupt(
                {
                    "question": "Review the following tool calls",
                    "tool_calls": [
                        {
                            "index": i,
                            "name": t['name'],
                            "args": t['args']
                        }
                        for i, t in enumerate(tool_calls)
                    ]
                }
            )
            action = answer.get("action")

            if action == "cancel":
                results = []
                for t in tool_calls:
                    message = ToolMessage(
                        content="Cancelled by user.",
                        tool_call_id=str(t["id"]),
                        name=t["name"]
                    )
                    results.append(message)
                return {
                    "messages": results,
                    "cancelled": True
                }
            
            if action == 'edit':
                index = answer["tool_index"]
                new_args = answer["args"]
                if not isinstance(index, int) or not 0 <= index < len(tool_calls):
                    raise ValueError("Invalid tool index")

                if not isinstance(new_args, dict):
                    raise ValueError("Tool arguments must be a dictionary")

                tool_calls[index] = {
                    **tool_calls[index],
                    "args": new_args
                }

                print("Tool arguments updated.")
                continue

            if action == "approve":
                break

            raise ValueError(f"Unknown HITL action: {action}")

        results = []
        for t in tool_calls:
            # print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ... bad tool name ...")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(
                ToolMessage(
                    tool_call_id=t['id'],
                    name=t['name'],
                    content=str(result)
                )
            )

        return {"messages": results}


    def exists_action(
        self,
        state: AgentState
    ) -> bool:

        result = state["messages"][-1]
        return len(result.tool_calls) > 0


    def route_after_action(
        self,
        state: AgentState
    ) -> bool:

        return state.get("cancelled", False)
        

