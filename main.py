from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
import re
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('langgraph_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# init model from ollama
model = init_chat_model(
    "ollama:gemma3:12b-it-qat",
)

logger.info("Model initialized successfully")

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

CRITICAL INSTRUCTIONS:
- Only use the exact function names shown above (no prefixes like 'weather_tool.' or 'tools.')
- Call functions EXACTLY as shown in the examples below
- When you need to call a function, wrap your function call in ```tool_code``` tags
- Use the exact function names: get_weather

CORRECT Examples:
```tool_code
get_weather("San Francisco")
```

```tool_code
get_weather("New York")
```

INCORRECT Examples (DO NOT USE):
- weather_tool.get_weather("San Francisco") ❌
- tools.get_weather("San Francisco") ❌
- weather.get_weather("San Francisco") ❌

User: {user_message}

Think step by step: Does this request require using one of the available functions? If yes, use the EXACT function name from the list above."""
    
    return prompt

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    logger.info(f"[ROUTER] Checking last message for tool code...")
    logger.info(f"[ROUTER] Last message content preview: {str(last_message.content)[:100]}...")
    
    # Check if the last message contains tool code
    if hasattr(last_message, 'content') and '```tool_code' in str(last_message.content):
        logger.info("[ROUTER] Tool code detected - routing to tools")
        return "tools"
    
    logger.info("[ROUTER] No tool code detected - routing to respond")
    return "respond"

def should_retry(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message contains a tool error
    if hasattr(last_message, 'content') and "Tool execution failed:" in str(last_message.content):
        logger.info("[RETRY] Tool error detected - allowing retry")
        return "think"
    
    logger.info("[RETRY] No error detected - proceeding to respond")
    return "respond"

def think(state: MessagesState):
    messages = state["messages"]
    logger.info(f"[THINK] Processing {len(messages)} messages")
    
    # For the first message, create a tool-aware prompt
    if len(messages) == 1:
        user_message = messages[0].content
        logger.info(f"[THINK] First message - creating tool-aware prompt for: {user_message}")
        prompt = create_tool_prompt(user_message)
        response = model.invoke([{"role": "user", "content": prompt}])
    else:
        logger.info("[THINK] Continuing conversation with existing context")
        response = model.invoke(messages)
    
    logger.info(f"[THINK] Model response: {response.content[:200]}...")
    return {"messages": [response]}

def execute_tools(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    logger.info("[TOOLS] Executing tool calls...")
    logger.info(f"[TOOLS] Last message content: {last_message.content}")
    
    # Extract and execute tool calls
    tool_output = extract_tool_calls(last_message.content)
    
    if tool_output:
        logger.info(f"[TOOLS] Tool execution result: {tool_output}")
        # Extract just the result from tool_output
        result_match = re.search(r'```tool_output\n(.*?)\n```', tool_output, re.DOTALL)
        if result_match:
            clean_result = result_match.group(1).strip()
            logger.info(f"[TOOLS] Clean result: {clean_result}")
            
            # Check if there was an error and provide helpful feedback
            if "Error:" in clean_result:
                from langchain_core.messages import AIMessage
                error_feedback = f"""Tool execution failed: {clean_result}

If you tried to call a function, remember to use the exact function names:
- get_weather("location") ✓
- NOT weather_tool.get_weather("location") ✗

Available functions: {', '.join(TOOLS.keys())}"""
                response = AIMessage(content=error_feedback)
                logger.warning(f"[TOOLS] Tool error occurred: {clean_result}")
            else:
                from langchain_core.messages import AIMessage
                response = AIMessage(content=f"Tool result: {clean_result}")
                logger.info("[TOOLS] Tool executed successfully")
            
            return {"messages": [response]}
    
    logger.warning("[TOOLS] No tool output generated")
    return {"messages": []}

def respond(state: MessagesState):
    messages = state["messages"]
    
    logger.info(f"[RESPOND] Generating final response from {len(messages)} messages")
    
    # Generate a clean response for the user based on the conversation
    prompt = """Based on the conversation and any tool results, provide a clear, helpful response to the user. 
    Do not include any tool code or internal reasoning. Just give a direct, conversational answer.
    If there are tool results, incorporate them naturally into your response."""
    
    # Add the prompt as a system message and get response
    conversation_with_prompt = messages + [{"role": "user", "content": prompt}]
    response = model.invoke(conversation_with_prompt)
    
    logger.info(f"[RESPOND] Final response: {response.content}")
    return {"messages": [response]}

# Build the LangGraph workflow
builder = StateGraph(MessagesState)
builder.add_node("think", think)
builder.add_node("tools", execute_tools)
builder.add_node("respond", respond)

builder.add_edge(START, "think")
builder.add_conditional_edges("think", should_continue, ["tools", "respond"])
builder.add_conditional_edges("tools", should_retry, ["think", "respond"])
builder.add_edge("respond", END)
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

# Interactive conversation loop with context preservation
def run_conversation_loop():
    conversation_history = []
    
    logger.info("Starting interactive conversation loop")
    print("=== INTERACTIVE CONVERSATION (type 'quit' to exit) ===")
    
    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            logger.info("User requested exit")
            print("Goodbye!")
            break
        
        logger.info(f"[USER INPUT] {user_input}")
        
        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))
        
        # Invoke graph with full conversation history
        logger.info(f"[GRAPH] Invoking graph with {len(conversation_history)} messages")
        result = graph.invoke({"messages": conversation_history})
        
        # Update conversation history with the assistant's response
        conversation_history.extend(result["messages"])
        
        # Print only the latest assistant response
        latest_response = result["messages"][-1]
        logger.info(f"[FINAL OUTPUT] {latest_response.content}")
        print(f"Assistant: {latest_response.content}")

# Run the conversation loop
if __name__ == "__main__":
    logger.info("Application started")
    run_conversation_loop()
    logger.info("Application ended")