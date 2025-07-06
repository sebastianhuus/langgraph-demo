# Tool-Augmented Conversational AI with Gemma and LangGraph

This Jupyter Notebook demonstrates a conversational AI workflow using the ReAct (Reasoning and Acting) paradigm, powered by [LangChain](https://python.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph). The assistant can answer user queries, invoke Python tools when needed, and generate natural, helpful responses.

## Why
Gemma models are widely recognized as some of the best small language models available in 2025, offering strong performance and the ability to run on most devices. However, Gemma does not natively support tool calling, which limits its usefulness for automation and real-world tasks that require access to external tools.

This notebook provides a practical framework for enabling tool-augmented conversations with Gemma models. Inspired by [Tech with Tim's YouTube tutorial](https://youtu.be/1w5cCXlh7JQ?si=oIKND7Gqy87fKtdR), it extends the basic approach to support robust tool invocation and integration. The goal is to make Gemma a more capable assistant for automation and workflow tasks by bridging the gap between language modeling and real tool access.

[Alexander Galea also had an interesting take on this](https://github.com/zazencodes/zazencodes-season-2/tree/main), but he used cloud models with native tool calling, which most other resources also do. Therefore, I needed to figure out a different way to implement this system so I can use it for sensitive enterprise workflows.

## Features

- **Tool-Augmented Responses:**  
  The assistant can call Python functions (tools) such as weather lookup, news retrieval, stock price queries, and smart home actions.

- **ReAct Agent Loop:**  
  The agent reasons about when to use tools and how to incorporate their outputs into its replies.

- **Prompt Engineering for Gemma:**  
  Since Gemma models do **not** support native tool calling, the notebook uses explicit, instruction-based prompting and manual parsing to achieve tool-augmented responses.

- **Automated Testing:**  
  Includes automated tests to verify that the agent correctly invokes tools and produces expected outputs for various prompts.

- **Stress Testing:**  
  Evaluates the model's ability to handle a large set of available tools and maintain accuracy.

## How It Works

1. **Prompting Strategy:**  
   - The system prompt includes detailed docstrings and function signatures for each available tool.
   - The model is instructed to wrap tool calls in special code blocks (````tool_code````), making them easy to detect and parse.

2. **Manual Tool Call Extraction:**  
   - After the model generates a response, regular expressions extract any code within ````tool_code```` blocks.
   - This code is executed in a controlled environment, and the output is wrapped in a ````tool_output```` block.

3. **Response Generation:**  
   - The agent incorporates tool outputs into its final response, ensuring natural and helpful replies.

4. **Testing and Evaluation:**  
   - The notebook includes both single-shot and automated multi-prompt tests to assess tool invocation accuracy and response quality.

## Model Support

- Supports both Gemma3 4B and 12B models. 27B model probably works best but I don't have the hardware to run it.
- Uses [Ollama](https://ollama.com/) for local model inference.

## Usage

1. **Install dependencies:**  
Make sure you have Python 3.12 or lower and the required packages from [pyproject.toml](pyproject.toml):

```sh
uv pip install -r pyproject.toml
```

Currently, you can't use Python 3.13 which is a new one because it's not compatible with the Langchain libraries at time of writing.

2. **Start Ollama and pull the required Gemma models.**

3. **Run the notebook:**  
   Open [tool-calling.ipynb](tool-calling.ipynb) in Jupyter or VS Code and execute the cells.

4. **Review results:**  
   - Inspect the conversation flow and tool usage in the output.
   - Check the automated test results for model/tool accuracy.

## Advanced Topics

- **Hierarchical Tool Management:**  
  The notebook discusses strategies for scaling to large toolsets, such as virtual service calls and progressive tool disclosure.

- **Prompt Engineering:**  
  Includes best practices for instructing models to use tools reliably, even without native function-calling support.

## Acknowledgements

- [davidmuraya/gemma3 main.py](https://github.com/davidmuraya/gemma3/blob/main/main.py)
- [Philipp Schmid's blog on Gemma function calling](https://www.philschmid.de/gemma-function-calling)
- [Reddit: PSA on Gemma 3 QAT GGUF Models](https://www.reddit.com/r/LocalLLaMA/comments/1jvi860/psa_gemma_3_qat_gguf_models_have_some_wrongly/)
- [Tech with Tim's YouTube tutorial](https://youtu.be/1w5cCXlh7JQ?si=oIKND7Gqy87fKtdR)
- [Alexander Galea's ZazenCodes Season 2](https://github.com/zazencodes/zazencodes-season-2/tree/main)

