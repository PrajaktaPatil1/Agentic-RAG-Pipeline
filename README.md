# Agentic-RAG-Pipeline
 This project demonstrates an **Agentic Retrieval-Augmented Generation (RAG)** pipeline built using **LangChain**, showcasing how to integrate multiple tools with an LLM to enable intelligent, dynamic reasoning workflows.

 ##  What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines:
- **Retrieval**: Search for relevant context from a knowledge base (e.g., documents, databases, APIs).
- **Generation**: Use an LLM to generate answers using the retrieved context.

### Traditional RAG:
- Works in a simple loop: input → retrieve documents → generate response.
- LLM is mostly passive and relies heavily on retrieved text.

---

##  What is Agentic RAG?

**Agentic RAG** takes RAG a step further. It treats the LLM as an *active agent* that:
- **Plans** what tools to use.
- **Decides** when to retrieve, summarize, calculate, or call APIs.
- **Reasons** over results and composes a final answer.

It combines:
- Tool usage (search, calculator, code interpreter, API callers, etc.)
- Memory or intermediate steps
- Planning via agents

> In short: LLM becomes a multi-step decision-maker, not just a text generator.

---

##  What This Pipeline Does

This pipeline uses **LangChain** to:
- Create an **agent-based RAG architecture**
- Connect to **multiple tools** like:
  - Web search
  - Calculator
  - File retriever (PDFs, text, etc.)
  - SQL/NoSQL DBs
- Execute **multi-step reasoning**: LLM decides what tools to use, retrieves knowledge, combines insights, and answers smartly.

 What This Pipeline Does
This pipeline uses LangChain to build an Agentic RAG (Retrieval-Augmented Generation) system with intelligent routing and tool use. Specifically, it enables:

Use of a LangGraph agent that dynamically chooses what to do with a given user input.

Integration of a classification tool to route questions into categories like Constitution, LLM, or Latest topics.

Conditional logic to fetch documents, embed them, and perform retrieval for selected categories.

Invocation of a Google Gemini 1.5 Flash LLM to generate answers.

Embedding of documents using BAAI/bge-small-en from HuggingFace.

Vector-based retrieval using FAISS.

Document loading from PDFs using PyPDFLoader.

 Tech Stack & Tools Used
 LLM (Large Language Model)
Google Gemini 1.5 Flash via langchain_google_genai.ChatGoogleGenerativeAI

Tools Used
LangChain Agents & LangGraph for multi-step agent execution and state management

Topic Classifier Tool using PydanticOutputParser and prompt templates

PDF File Retriever using PyPDFLoader

Embedding Tool using HuggingFaceEmbeddings with BAAI/bge-small-en

Vector Store: FAISS for fast similarity-based document retrieval

Supporting Libraries
pydantic for structured response parsing

operator, typing for custom agent state definitions

RecursiveCharacterTextSplitter to chunk documents

langchain_core, langgraph, langchain.output_parsers to handle core workflow logic

