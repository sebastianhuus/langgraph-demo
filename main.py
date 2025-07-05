from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
import re

# init model from ollama
model = init_chat_model(
    "ollama:gemma3:4b",
    n_ctx=4096,
)

def get_weather(location: str):
    """Returns weather info for a location."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

# Available tools dictionary
TOOLS = {
    "get_weather": get_weather
}

def extract_tool_calls(text):
    """Extract tool calls from model output using regex parsing."""
    pattern = r"```tool_code\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        try:
            # Execute the tool call safely
            result = eval(code, {"__builtins__": {}}, TOOLS)
            return f'```tool_output\n{result}\n```'
        except Exception as e:
            return f'```tool_output\nError: {str(e)}\n```'
    return None

def create_tool_prompt(user_message):
    """Create a prompt that instructs Gemma3 to use tools."""
    tool_definitions = []
    for name, func in TOOLS.items():
        tool_definitions.append(f"def {name}({func.__code__.co_varnames[0]}: str) -> str:\n    \"\"\"{func.__doc__}\"\"\"")
    
    tools_str = "\n".join(tool_definitions)
    
    prompt = f"""You are a helpful assistant with access to predefined Python functions. Think step by step why and how these functions should be used.

Available functions:
```python
{tools_str}
```

Instructions:
- Only use the predefined functions listed above
- When you need to call a function, wrap your function call in ```tool_code``` tags
- Think about whether the user's request requires using a function
- If a function is needed, generate the appropriate function call
- Do not invent or hallucinate additional information

Example:
```tool_code
get_weather("San Francisco")
```

User: {user_message}

Think step by step: Does this request require using one of the available functions? If yes, which function and with what parameters?"""
    
    return prompt

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    # Check if the last message contains tool code
    if hasattr(last_message, 'content') and '```tool_code' in str(last_message.content):
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    # For the first message, create a tool-aware prompt
    if len(messages) == 1:
        user_message = messages[0].content
        prompt = create_tool_prompt(user_message)
        response = model.invoke([{"role": "user", "content": prompt}])
    else:
        response = model.invoke(messages)
    return {"messages": [response]}

def execute_tools(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Extract and execute tool calls
    tool_output = extract_tool_calls(last_message.content)
    
    if tool_output:
        # Extract just the result from tool_output
        result_match = re.search(r'```tool_output\n(.*?)\n```', tool_output, re.DOTALL)
        if result_match:
            clean_result = result_match.group(1).strip()
            # Create a direct response with just the tool result
            from langchain_core.messages import AIMessage
            response = AIMessage(content=f"The weather information: {clean_result}")
            return {"messages": [response]}
    
    return {"messages": []}

# Build the LangGraph workflow
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", execute_tools)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", END)
graph = builder.compile()

# Run the agent
def print_conversation(result):
    print("=== CONVERSATION FLOW ===")
    messages = result["messages"]
    
    for i, message in enumerate(messages):
        print(f"\n--- Message {i+1} ---")
        print(f"Type: {type(message).__name__}")
        print(f"Content: {message.content}")
        
        # Check if this message contains a tool call
        if '```tool_code' in str(message.content):
            print("🔧 TOOL CALL DETECTED")
            tool_output = extract_tool_calls(message.content)
            if tool_output:
                print(f"Tool Result: {tool_output}")

result = graph.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})
print_conversation(result)
result = graph.invoke({"messages": [HumanMessage(content="What's the weather in Oslo, Norway?")]})
print_conversation(result)