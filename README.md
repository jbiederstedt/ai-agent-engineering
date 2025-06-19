# AI Agent Engineering with LangGraph

This repository contains the practical implementations and exercises from the **[2025 Bootcamp: Understand and Build Professional AI Agents](https://www.udemy.com/course/2025-bootcamp-understand-and-build-professional-ai-agents/)** online course by Julio Colomer.

The course covers advanced concepts in AI agent development using LangGraph, LangChain, and modern AI technologies.

## Project Overview

This repository contains a comprehensive collection of AI agent implementations, covering fundamental to advanced concepts in LangGraph development. The project includes both Python scripts and Jupyter notebooks that demonstrate various AI agent patterns and architectures.

## 🚀 Features

- **Basic Graph Operations**: Simple state management and node transitions
- **Advanced Agent Patterns**: Memory management, routing, and conditional logic
- **Schema Management**: Pydantic integration for structured data handling
- **Memory Systems**: Short-term and long-term memory implementations
- **Streaming & Debugging**: Real-time processing and debugging capabilities
- **Parallelization**: Multi-threaded and concurrent processing
- **Research Assistant**: Complete research workflow implementation
- **Human-in-the-Loop**: Interactive feedback and human oversight

## 📁 Project Structure

### Python Scripts (Sequential Learning)

- `003-basic-graph.py` - Basic LangGraph setup and state management
- `004-graph-with-chain.py` - Integration with LangChain components
- `005-graph-with-router.py` - Conditional routing and decision making
- `006-basic-agent.py` - Simple agent implementation
- `007-agent-with-memory.py` - Memory-enabled agents
- `009-schema-with-pydantic.py` - Structured data with Pydantic
- `010-reducers.py` - State reduction patterns
- `011-primary-and-secondary-schemas.py` - Advanced schema management
- `012-filter-trim-messages.py` - Message processing and filtering
- `013-summarizing-messages.py` - Content summarization workflows
- `014-external-memory.py` - External memory storage
- `015-streaming.py` - Real-time streaming capabilities
- `016-breakpoints.py` - Debugging and breakpoint management
- `017-add-human-feedback.py` - Human-in-the-loop interactions
- `018-dynamic-breakpoints.py` - Dynamic debugging controls
- `019-debugging.py` - Advanced debugging techniques
- `020-parallelization.py` - Parallel processing implementations
- `021-subgraphs.py` - Modular graph composition
- `022-map-reduce.py` - Map-reduce patterns for large-scale processing
- `023-research-assistant.py` - Complete research workflow agent
- `025-store.py` - Data persistence and storage
- `026-profile-schema.py` - User profile management
- `027-collection-schema.py` - Collection-based data structures
- `028-agent-with-LT-memory.py` - Long-term memory systems

### Jupyter Notebooks

Comprehensive notebooks (prefixed with `zzz-nb`) that provide detailed explanations, visualizations, and interactive examples for each concept.

### Additional Resources

- `graph001.png` through `graph004.png` - Visual representations of graph structures
- `state_db/` - State management database
- `tests/` - Test suite for the implementations

## 🛠️ Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- API keys for:
  - OpenAI
  - Tavily
  - LangSmith (optional, for tracing)

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai-agent-engineering
   ```

2. **Install dependencies using Poetry**

   ```bash
   poetry install
   ```

3. **Set up environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here

   # Optional: LangSmith for tracing
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=your_project_name
   ```

4. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

## 🎯 Getting Started

### Running Python Scripts

```bash
# Basic graph example
python 003-basic-graph.py

# Agent with memory
python 007-agent-with-memory.py

# Research assistant
python 023-research-assistant.py
```

### Running Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

## 📚 Learning Path

The project follows a progressive learning path:

1. **Fundamentals** (003-006): Basic graph concepts and simple agents
2. **Advanced Patterns** (007-014): Memory, schemas, and message processing
3. **Debugging & Control** (015-019): Streaming, breakpoints, and debugging
4. **Scalability** (020-022): Parallelization and subgraphs
5. **Real-world Applications** (023-028): Research assistants and production patterns

## 🔧 Key Technologies

- **LangGraph**: Graph-based AI agent framework
- **LangChain**: LLM application framework
- **OpenAI**: GPT models for natural language processing
- **Pydantic**: Data validation and settings management
- **Jupyter**: Interactive development and documentation
- **Poetry**: Dependency management and packaging
