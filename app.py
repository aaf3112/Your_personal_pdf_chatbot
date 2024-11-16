import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for faster PDF processing
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle  # For saving and loading vector store
from hashlib import md5
import time  # For performance tracking
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

# Define CSS for styling
css = """
    <style>
        .chat-container { display: flex; flex-direction: column; align-items: flex-start; max-width: 700px; margin: auto; padding: 20px; background-color: #f8f8f8; border-radius: 8px; }
        .chat-box { display: flex; flex-direction: column; width: 100%; background-color: #f0f0f0; padding: 15px; border-radius: 8px; max-height: 500px; overflow-y: auto; margin-bottom: 10px; }
        .user-message { background-color: #D6E4FF; color: black; padding: 10px; border-radius: 8px; margin: 5px 0; align-self: flex-end; max-width: 80%; word-wrap: break-word; }
        .bot-message { background-color: #E3E3E3; color: black; padding: 10px; border-radius: 8px; margin: 5px 0; align-self: flex-start; max-width: 80%; word-wrap: break-word; }
        .input-container { display: flex; width: 100%; margin-top: 10px; }
        .input-box { width: 90%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
        .send-button { width: 10%; padding: 10px; background-color: #4CAF50; border-radius: 8px; color: white; border: none; cursor: pointer; font-weight: bold; }
        .send-button:hover { background-color: #45a049; }
    </style>
"""

def load_huggingface_model(model_name="google/flan-t5-large"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=1500,  # Increase max length to allow for longer outputs
            max_new_tokens=1500  # Adjust as needed
        )

        # Wrap in LangChain's HuggingFacePipeline for LLMChain compatibility
        return HuggingFacePipeline(pipeline=hf_pipeline)

    except Exception as e:
        st.error(f"Failed to load Hugging Face model: {str(e)}")
        return None

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/msmarco-distilbert-base-v3')
except Exception as e:
    st.error(f"Error initializing embeddings: {str(e)}")

def extract_text_from_pdf(pdf_file):
    doc_text = ""
    metadata = {}
    try:
        with fitz.open("pdf", pdf_file.read()) as doc:
            # Extract metadata
            metadata = doc.metadata
            for page_num in range(doc.page_count):
                page_text = doc[page_num].get_text("text")
                if page_text:
                    doc_text += page_text
                else:
                    st.warning(f"Warning: Page {page_num + 1} of {pdf_file.name} is empty or unreadable.")
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {str(e)}")
    return doc_text, metadata

def get_text_chunks(text):
    # Larger chunks with overlap
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=3500, chunk_overlap=350)
    return text_splitter.split_text(text)

def vectorstore_cache_path(text_chunks):
    hash_id = md5("".join(text_chunks).encode()).hexdigest()
    return f"vectorstore_{hash_id}.pkl"

def get_vectorstore(text_chunks):
    cache_path = vectorstore_cache_path(text_chunks)
    if os.path.exists(cache_path):
        st.info("Loading vectorstore from cache.")
        return load_vectorstore(cache_path)

    st.info("Generating new vectorstore...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Print retrieved documents for debugging
    retrieved_docs = retriever.get_relevant_documents("What is house credit default?")
    st.write("Retrieved documents:", retrieved_docs)  # Debug the result

    save_vectorstore(vectorstore, cache_path)
    return vectorstore

def save_vectorstore(vectorstore, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(vectorstore, f)

def load_vectorstore(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def get_conversation_chain(vectorstore, hf_pipeline):
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
    retriever = vectorstore.as_retriever()

    # Use return_source_documents=True to see what documents are retrieved
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=hf_pipeline,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

def get_pdf_text_and_metadata(pdf_docs):
    start_time = time.time()
    all_text = ""
    all_metadata = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_text_from_pdf, pdf_docs))

    for text, metadata in results:
        all_text += text
        all_metadata.append(metadata)

    # Store metadata for future reference
    st.session_state.pdf_metadata = all_metadata  # Store metadata in session state

    st.info(f"PDF text extraction and metadata retrieval completed in {time.time() - start_time:.2f} seconds.")
    return all_text, all_metadata

def render_answer(answer_text):
    """Format long answers properly."""
    # Ensure multiple paragraphs are handled
    return answer_text.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")

def handle_userinput():
    if "user_question" not in st.session_state or not st.session_state.user_question:
        return

    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.error("Conversation chain is not initialized. Please process documents first.")
        return

    user_question = st.session_state.user_question
    conversation_chain = st.session_state.conversation

    # Check if the user is asking about metadata
    metadata = st.session_state.get("pdf_metadata", [])

    if "title" in user_question.lower():
        # Answer metadata question related to title
        if metadata:
            title = metadata[0].get("title", "Title not found")
            st.session_state.chat_history.append({"user": user_question, "bot": f"The title of the document is: {title}"})
        else:
            st.session_state.chat_history.append({"user": user_question, "bot": "No metadata available."})
        return

    if "author" in user_question.lower():
        # Answer metadata question related to author
        if metadata:
            author = metadata[0].get("author", "Author not found")
            st.session_state.chat_history.append({"user": user_question, "bot": f"The author of the document is: {author}"})
        else:
            st.session_state.chat_history.append({"user": user_question, "bot": "No metadata available."})
        return

    try:
        # Formulate a query that requests a full, detailed answer
        query = f"Please provide a full and detailed answer to this question based on the document: {user_question}"

        # Process the question using the conversation chain
        response = conversation_chain(query)
        answer_text = response.get('answer', '').strip()

        # Check if the answer is missing or incomplete
        if not answer_text or answer_text == "###":
            answer_text = "I'm sorry, I couldn't find a relevant answer."

        st.session_state.chat_history.append({"user": user_question, "bot": answer_text})
        st.session_state.user_question = ""

    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with your PDFs", page_icon="📂")
    st.write(css, unsafe_allow_html=True)

    if "hf_pipeline" not in st.session_state:
        st.session_state.hf_pipeline = load_huggingface_model()
        if st.session_state.hf_pipeline is None:
            return

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_chat_history" not in st.session_state:
        st.session_state.show_chat_history = False

    st.header("Chat with your PDFs 📂")

    user_question = st.text_input("Ask a question about your documents:", key="user_question")
    st.button("Send", on_click=handle_userinput)

    if st.button("View Chat History"):
        st.session_state.show_chat_history = not st.session_state.show_chat_history

    with st.container():
        st.write('<div class="chat-container">', unsafe_allow_html=True)
        st.write('<div class="chat-box">', unsafe_allow_html=True)

        if st.session_state.show_chat_history and st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.write(f'<div class="user-message"><b>Question:</b> {chat["user"]}</div>', unsafe_allow_html=True)
                st.write(f'<div class="bot-message"><b>Answer:</b> {chat["bot"]}</div>', unsafe_allow_html=True)
        elif not st.session_state.show_chat_history and st.session_state.chat_history:
            chat = st.session_state.chat_history[-1]
            st.write(f'<div class="user-message"><b>Question:</b> {chat["user"]}</div>', unsafe_allow_html=True)
            st.write(f'<div class="bot-message"><b>Answer:</b> {chat["bot"]}</div>', unsafe_allow_html=True)

        st.write('</div>', unsafe_allow_html=True)
        st.write('</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if pdf_docs:
            try:
                # Extract text and metadata
                raw_text, metadata = get_pdf_text_and_metadata(pdf_docs)

                # Get text chunks and generate vectorstore
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                # Initialize conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.hf_pipeline)

                st.success("Documents processed successfully! You can ask questions now.")

                # Instead of showing metadata, show a success message
                st.info("Metadata successfully extracted.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

if __name__ == "__main__":
    main()