from . import agent, prompt, tools
import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from IPython.display import Image
from PIL import Image as img

MODEL = "deepseek:deepseek-flash"

memory = InMemorySaver()

thread = {"configurable": {"thread_id": "1"}}

def cancel_pending(abot, thread):
    state = abot.graph.get_state(thread)
    last = state.values["messages"][-1]

    cancelled = [
        ToolMessage(
            content="Cancelled by user.",
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        for tc in last.tool_calls
    ]
    abot.graph.update_state(
        thread,
        {"messages": cancelled + [AIMessage(content="Okay, I cancelled that search.")]},
        as_node="llm",   # must match your LLM node's name in add_node(...)
    )


def edit_tool_calls(abot, thread):
    state = abot.graph.get_state(thread)
    last = state.values["messages"][-1]

    new_calls = []
    for tc in last.tool_calls:
        print(f"\nTool: {tc['name']}")
        print(f"Args: {json.dumps(tc['args'], ensure_ascii=False)}")
        raw = input("New args as JSON (Enter to keep): ").strip()

        args = tc["args"]
        if raw:
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                print("Invalid JSON, keeping the original args.")

        new_calls.append({**tc, "args": args})   # keep the same id and name

    new_msg = AIMessage(
        content=last.content,
        tool_calls=new_calls,
        id=last.id,          # same id means replace, not append
    )
    abot.graph.update_state(thread, {"messages": [new_msg]}, as_node="llm")


def main() -> None:

    model = init_chat_model(model=MODEL)
    abot = agent.Agent(model, [tools.tool], checkpointer=memory, system=prompt.system_prompt)

    print("""
                ==================================================
                |type your question                              |
                |if you want to end the conversation, type 'exit'|
                ==================================================
    """)
    
    while True:

        question = input("> ")

        if question == 'exit':
            print("Conversation has been done")
            break

        messages = [HumanMessage(content=question)]
        result = abot.graph.invoke({"messages": messages}, thread)
        for m in result["messages"]:
            m.pretty_print()

        while abot.graph.get_state(thread).next:
            state = abot.graph.get_state(thread)

            interrupt_data = state.tasks[0].interrupts[0].value

            print("\n========== HUMAN REVIEW ==========")
            print(interrupt_data["question"])

            for tool in interrupt_data["tool_calls"]:
                print(f"\nTool: {tool['name']}")
                print(f"Args: {json.dumps(tool['args'], indent=2, ensure_ascii=False)}")
            print("\nActions:")
            print("1. Approve")
            print("2. Edit")
            print("3. Cancel")

            choice = input("> ").strip()

            if choice == "1":
                answer = {"action": "approve"}

            elif choice == "2":

                index = int(input("Tool index to edit: "))

                raw = input("New arguments as JSON: ")

                try:
                    new_args = json.loads(raw)
                except json.JSONDecodeError:
                    print("Invalid JSON.")
                    continue    # then fall through and run

                answer = {
                    "action": "edit",
                    "tool_index": index,
                    "args": new_args
                }

            elif choice == "3":
                answer = {"action": "cancel"}

            else:
                print("Invalid choice.")
                continue

            result = abot.graph.invoke(Command(resume=answer), thread)
            if result.get("messages"):
                result["messages"][-1].pretty_print()

            # result["messages"][-1].pretty_print()





def test() -> None:
    from langgraph.graph import StateGraph, START
    from langgraph.checkpoint.memory import InMemorySaver
    from typing_extensions import TypedDict, NotRequired
    from langchain_core.utils.uuid import uuid7
    from langgraph.types import interrupt, Command

    class State(TypedDict):
        value: list[str]

    def ask_human(state: State):
        answer = interrupt("What is your name?")
        return {"value": [f"Hello, {answer}!"]}

    def final_step(state: State):
        return {"value": ["Done"]}

    graph = (
        StateGraph(State)
        .add_node("ask_human", ask_human)
        .add_node("final_step", final_step)
        .add_edge(START, "ask_human")
        .add_edge("ask_human", "final_step")
        .compile(checkpointer=InMemorySaver())
    )

    config = {"configurable": {"thread_id": "1"}}

    # First run: hits interrupt
    graph.invoke({"value": []}, config)
    # Resume with answer
    graph.invoke(Command(resume="Alice"), config)

    # Replay from before ask_human
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    replay_result = graph.invoke(None, before_ask.config)
    # Pauses at interrupt — waiting for new Command(resume=...)

    # Fork from before ask_human
    fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    # Pauses at interrupt — waiting for new Command(resume=...)

    # Resume the forked interrupt with a different answer
    graph.invoke(Command(resume="Bob"), fork_config)
    # Result: {"value": ["forked", "Hello, Bob!", "Done"]}

    for state in history:
        print(f"next={state.next}, checkpoint_id={state.config['configurable']['checkpoint_id']}")