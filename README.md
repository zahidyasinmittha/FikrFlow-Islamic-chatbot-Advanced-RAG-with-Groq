# FikrFlow

**FikrFlow** is an advanced retrieval-augmented generation (RAG) chatbot designed to provide authentic Islamic knowledge. Leveraging cutting-edge language models, hybrid retrieval techniques (vector + BM25), and a modern conversational interface built with Streamlit, FikrFlow offers users a powerful way to explore and discuss Islamic teachings.

A live demonstration of the chatbot is available at: [https://fikrflow.streamlit.app/](https://fikrflow.streamlit.app/)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [How to Run It on Your Own Machine](#how-to-run-it-on-your-own-machine)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Hybrid Retrieval:** Combines vector search using Chroma with BM25 lexical search for robust document retrieval.
- **Conversational Memory:** Maintains a full conversation history so users can refer back to previous exchanges.
- **Prompt Engineering:** Constructs detailed prompts incorporating conversation context and relevant documents.
- **Streamlit Interface:** Provides a clean, responsive chat interface with customizable CSS.
- **Groq LLM Integration:** Leverages Groq's API for generating detailed, context-aware responses.
- **Islamic Focus:** Curated content and retrieval methods tailored for authentic Islamic knowledge.

---

## Installation

### 1. Clone the Repository

Clone the repository from GitHub:

```bash
git clone https://github.com/zahidyasinmittha/FikrFlow-Islamic-chatbot-Advanced-RAG-with-Groq.git
cd FikrFlow-Islamic-chatbot-Advanced-RAG-with-Groq

[Download chroma_db_islamic_text Folder](https://drive.google.com/drive/folders/1EsZ3SuS_z_vXUZFhDJanQa_8l51_-bg-)
