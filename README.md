# Your_personal_pdf_chatbot
This repository provides a Content Engine for analyzing and comparing multiple PDF documents. Built using LangChain, Streamlit, and Hugging Face, this system leverages Retrieval-Augmented Generation (RAG) to retrieve, assess, and generate insights from documents.
## Features

- *Document Parsing & Comparison*  
   - Extracts and analyzes text from multiple PDF documents.
   - Identifies and highlights differences between documents, such as financial figures, risk factors, and business descriptions.

- *Vector Store Ingestion*  
   - Uses embeddings to convert document content into vectors for efficient querying.
   - Vectors are stored in a FAISS vector store for rapid retrieval.

- *Query Engine*  
   - A powerful query engine based on RAG, enabling users to ask detailed questions about the documents.
   - Retrieves the most relevant document content to generate informed responses.

- *Local Language Model*  
   - Utilizes a local instance of a Hugging Face language model to generate answers and insights directly from the document content.
   - Operates independently without requiring external API calls.

- *Chatbot Interface*  
   - An interactive conversational interface built with Streamlit, allowing users to ask questions about the documents in a chat-based format.

## Setup

1. *Clone the repository:*
     ```bash
   git clone https://github.com/aaf3112/Your_personal_pdf_chatbot.git
   cd Your_personal_pdf_chatbot

2. **Install dependencies:**
     ```bash
   pip install -r requirements.txt

3. *Run the application:*
   ```bash
   streamlit run app.py
   
## Technical Stack

- *Backend Framework*: [LangChain](https://github.com/hwchase17/langchain) – A toolkit for building LLM applications with a focus on retrieval-augmented generation (RAG).
- *Frontend Framework*: [Streamlit](https://streamlit.io/) – For building the web interface and user interaction.
- *Vector Store*: [FAISS](https://faiss.ai/) – For efficiently storing and querying document embeddings.
- *Embedding Model*: [Sentence-Transformers](https://www.sbert.net/) from Hugging Face – A local embedding model used to generate vector representations of document content.
- *Local LLM*: [Hugging Face's DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium) – A local model for generating answers and insights from document content.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain) for the powerful toolkit to build the retrieval system.
- [Streamlit](https://streamlit.io/) for simplifying the creation of web interfaces.
- [Hugging Face](https://huggingface.co/) for their transformer models and tools.
- [PyMuPDF](https://pymupdf.readthedocs.io/) for efficient PDF text extraction.
