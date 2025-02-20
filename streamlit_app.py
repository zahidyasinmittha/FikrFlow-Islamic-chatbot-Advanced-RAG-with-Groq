import os
from typing import List, Any, ClassVar, Optional
import pysqlite3 as sqlite3
import sys
sys.modules["sqlite3"] = sqlite3
import torch
import torch
torch.classes.__path__ = []
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain.schema import BaseRetriever, Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import gdown
from langchain.llms.base import LLM

# Groq LLM client
from groq import Groq  # pip install groq
import streamlit as st

def download_drive_folder_if_not_exists(folder_path: str, folder_url: str):
    """
    Downloads a folder from Google Drive using gdown if it does not exist locally.
    
    Parameters:
    - folder_path: The local folder path where the files should be stored.
    - folder_url: The public Google Drive folder URL.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        st.info("Downloading database folder from Google Drive...")
        # gdown.download_folder downloads files into the given output directory.
        gdown.download_folder(url=folder_url, output=folder_path, quiet=False, use_cookies=False)
        st.success("Download complete.")
    else:
        st.info("Database folder already exists. Skipping download.")

try:
    from rank_bm25 import BM25Okapi
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize
    BM25_ENABLED = True
except ImportError:
    BM25_ENABLED = False
    print("[WARNING] 'rank_bm25' and/or 'nltk' not installed. Hybrid Retrieval won't work.")

class ConversationMemory:
    """
    A simple class to wrap conversation history.
    """
    def __init__(self, history=None):
        self.conversation_history = history if history is not None else []

    def add_to_history(self, user_query: str, response: str):
        self.conversation_history.append({
            "user": user_query,
            "assistant": response
        })

    def get_context_as_text(self) -> str:
        return "\n".join(
            f"User: {turn['user']}\nAssistant: {turn['assistant']}" 
            for turn in self.conversation_history
        )
class HybridRetriever:
    """
    Demonstrates a hybrid approach: BM25 lexical + Chroma vector search.
    """
    def __init__(self, chroma_vectorstore: Chroma, documents: Optional[List[str]] = None):
        """
        :param chroma_vectorstore: An existing Chroma vector store.
        :param documents: (Optional) A list of raw doc texts for BM25. 
                          Only needed if you want lexical retrieval in addition to vector.
        """
        self.vectorstore = chroma_vectorstore
        self.bm25_enabled = False

        if documents and BM25_ENABLED:
            self.bm25_enabled = True
            tokenized_docs = [word_tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            self.documents = documents

    def hybrid_search(self, query: str, top_k: int = 5) -> List[str]:
        """
        1) Vector search from Chroma
        2) If BM25 is enabled, also do lexical search
        3) Merge results, remove duplicates, return top_k
        """
        # Vector-based retrieval
        vector_hits = self.vectorstore.similarity_search(query, k=top_k)
        vector_docs = [hit.page_content for hit in vector_hits]

        if not self.bm25_enabled:
            # If BM25 not set, just return vector results
            return vector_docs

        # BM25 lexical retrieval
        bm25_scores = self.bm25.get_scores(word_tokenize(query))
        scored_pairs = list(zip(self.documents, bm25_scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        bm25_docs = [sp[0] for sp in scored_pairs[:top_k]]

        # Combine & deduplicate
        combined = []
        seen = set()
        for doc in bm25_docs + vector_docs:
            if doc not in seen:
                seen.add(doc)
                combined.append(doc)

        return combined[:top_k]
    
class AugmentationModule:
    """
    Filters or structures retrieved texts before final generation.
    """
    @staticmethod
    def augment(doc_texts: List[str]) -> List[str]:
        """
        Example: remove duplicates, do minimal cleaning.
        """
        unique_texts = []
        seen = set()
        for text in doc_texts:
            cleaned = text.strip()
            if cleaned not in seen:
                seen.add(cleaned)
                unique_texts.append(cleaned)
        return unique_texts

class MultiStepRetriever:
    """
    Simple example: if 'hadith' appears in the query, we expand with 'sunnah', etc.
    Then we do a hybrid search (BM25 + vector), or fallback to just vector if BM25 not enabled.
    """
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid_retriever = hybrid_retriever

    def multi_step_search(self, query: str, top_k: int = 5) -> List[str]:
        expansions = []
        if "hadith" in query.lower():
            expansions.append("sunnah")

        if expansions:
            query_expanded = f"{query} {' '.join(expansions)}"
        else:
            query_expanded = query

        # Now call the hybrid search
        return self.hybrid_retriever.hybrid_search(query_expanded, top_k=top_k)

class PromptEngineer:
    """
    Combines conversation memory, relevant docs, and user query into one final prompt.
    """
    @staticmethod
    def build_prompt(user_query: str, conversation_context: str, relevant_docs: List[str]) -> str:
        prompt_parts = []

        if conversation_context:
            prompt_parts.append(f"Conversation Context:\n{conversation_context}")

        if relevant_docs:
            docs_str = "\n\n".join(f"Document:\n{doc}" for doc in relevant_docs)
            prompt_parts.append(f"Relevant Documents:\n{docs_str}")

        prompt_parts.append(f"User Query:\n{user_query}")
        prompt_parts.append("Provide a detailed answer, referencing the documents above as needed.")

        return "\n\n".join(prompt_parts)

class CrossEncoderReranker:
    """
    Custom cross-encoder reranker that:
      - Takes a query + retrieved documents,
      - Scores each document using a cross-encoder,
      - Returns the top K documents.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.model.to(self.device)

    def rerank(self, query: str, docs: list, top_k=3):
        inputs = []
        for d in docs:
            # If it's a LangChain Document, use `.page_content`
            doc_text = d.page_content if hasattr(d, "page_content") else d
            inputs.append((query, doc_text))

        encoded = self.tokenizer(
            [q for q, _ in inputs],
            [d for _, d in inputs],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            scores = outputs.logits.squeeze(-1)

        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _score in scored_docs[:top_k]]
        return top_docs
class AdvancedRetriever(BaseRetriever):
    """
    A custom two-stage retriever that:
      1) Performs approximate vector search (top_n).
      2) Re-ranks the results via a cross-encoder (final_k docs).
    """
    class Config:
        extra = 'allow'  # Use literal value as recommended in Pydantic V2

    vector_store: Any
    reranker: Any
    top_n: int = 10
    final_k: int = 2

    def _get_relevant_documents(self, query: str) -> List[Document]:
        initial_docs = self.vector_store.similarity_search(query, k=self.top_n)
        top_docs = self.reranker.rerank(query, initial_docs, top_k=self.final_k)
        return top_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Same as above, but async
        initial_docs = self.vector_store.similarity_search(query, k=self.top_n)
        top_docs = self.reranker.rerank(query, initial_docs, top_k=self.final_k)
        return top_docs

class GroqLLM(LLM):
    """
    Custom LLM wrapper to integrate with Groq's API.
    """
    client: ClassVar[Any] = None  # Marked as ClassVar so it's not treated as a field
    model: str = None

    class Config:
        extra = 'allow'

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.__class__.client = Groq(api_key=api_key)
        self.model = model

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.__class__.client.chat.completions.create(
            messages=messages,
            model=self.model
        )
        return response.choices[0].message.content

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

def main():
    ###################################################
    # 1) PAGE CONFIG & CUSTOM CSS
    ###################################################
    st.set_page_config(
        page_title="FikrFlow",
        page_icon="⚡",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    # Example usage in your main code:
    db_folder = "./chroma_db_islamic_text"
    drive_folder_url = "https://drive.google.com/drive/folders/1EsZ3SuS_z_vXUZFhDJanQa_8l51_-bg-?usp=sharing"
    download_drive_folder_if_not_exists(db_folder, drive_folder_url)

    custom_css = """
    <style>
    /* Body gradient background */
    body {
        background: linear-gradient(to right top, #f2f2f2, #e1e8f0) !important;
    }
    /* Transparent main container */
    [data-testid="stAppViewContainer"] > .main {
        background-color: transparent !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid #ccc;
        padding: 1rem;
    }
    /* Title and header styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: "Trebuchet MS", sans-serif;
        color: #2F3A56;
    }
    /* Input label styling */
    .stTextInput label {
        font-weight: 600;
        color: #333;
    }
    /* Button styling */
    div.stButton > button {
        color: #fff;
        background-color: #6C63FF;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #5952d4;
        cursor: pointer;
    }
    /* Chat container styling */
    .chat-container {
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    /* User message styling */
    .chat-message.user {
        background-color: #89d6e6;
        border: 1px solid #b3e5fc;
        border-radius: 10px;
        padding: 10px;
        max-width: 60%;
        align-self: flex-end;
        text-align: right;
        margin-bottom: 5px;
    }
    /* Assistant message styling */
    .chat-message.assistant {
        background-color: #06447f;
        border: 1px solid #d0d0d0;
        border-radius: 10px;
        padding: 10px;
        max-width: 80%;
        align-self: flex-start;
        text-align: left;
        margin-bottom: 10px;
        color: white;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    ###################################################
    # 2) INITIALIZE CONVERSATION HISTORY (PERSISTENT)
    ###################################################
    # Use a list to store all Q&A pairs; create it only once.
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    
    ###################################################
    # 3) APP TITLE
    ###################################################
    st.title("FikrFlow")
    
    ###################################################
    # 4) INITIALIZE OTHER COMPONENTS (UNCHANGED)
    ###################################################
    if "vectorstore" not in st.session_state:
        CHROMA_DIR = "./chroma_db_islamic_text"  # Adjust if needed
        if not os.path.exists(CHROMA_DIR):
            st.error(f"Could not find Chroma DB directory: {CHROMA_DIR}")
            return
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
        st.session_state["vectorstore"] = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embed_model
        )
    if "documents" not in st.session_state:
        st.session_state["documents"] = []
    if "hybrid_retriever" not in st.session_state:
        st.session_state["hybrid_retriever"] = HybridRetriever(
            chroma_vectorstore=st.session_state["vectorstore"],
            documents=st.session_state["documents"]
        )
    if "multi_step_retriever" not in st.session_state:
        st.session_state["multi_step_retriever"] = MultiStepRetriever(
            st.session_state["hybrid_retriever"]
        )
    if "reranker" not in st.session_state:
        cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        st.session_state["reranker"] = CrossEncoderReranker(model_name=cross_encoder_model)
    if "augmentation_module" not in st.session_state:
        st.session_state["augmentation_module"] = AugmentationModule()
    if "prompt_engineer" not in st.session_state:
        st.session_state["prompt_engineer"] = PromptEngineer()
    if "groq_llm" not in st.session_state:
        GROQ_API_KEY = "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"  # Replace with your key
        GROQ_MODEL_NAME = "llama-3.3-70b-versatile"  # Example model name
        st.session_state["groq_llm"] = GroqLLM(model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY)
    
    ###################################################
    # 5) SIDEBAR CONTROLS
    ###################################################
    st.sidebar.title("Chat Controls")
    st.sidebar.image("1.webp", use_container_width=True)
    st.sidebar.markdown("### Welcome to the FikrFlow App")
    st.sidebar.markdown("**FikrFlow** is your gateway to authentic Islamic knowledge.")
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Chat"):
        st.session_state["conversation_history"] = []
        st.success("Chat history has been reset!")
    
    ###################################################
    # 6) DISPLAY CONVERSATION HISTORY (AT THE TOP)
    ###################################################
    history_placeholder = st.empty()
    with history_placeholder.container():
        st.markdown("### Conversation History")
        if st.session_state["conversation_history"]:
            for i, turn in enumerate(st.session_state["conversation_history"], start=1):
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="chat-message user">
                            <strong>Q{i}:</strong> {turn['user']}
                        </div>
                        <div class="chat-message assistant">
                            <strong>A{i}:</strong> {turn['assistant']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No conversation yet. Ask a question below!")
    
    ###################################################
    # 7) USER INPUT SECTION (SEARCH BAR AT THE BOTTOM)
    ###################################################
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Enter your question:")
        submit_button = st.form_submit_button("Send Query")
    
    if submit_button:
        if user_input.strip():
            # Build conversation context from full history (if needed)
            conversation_context = "\n".join(
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in st.session_state["conversation_history"]
            )
            # (Perform multi-step retrieval, augmentation, re-ranking, prompt engineering, etc.)
            multi_step_docs = st.session_state["multi_step_retriever"].multi_step_search(user_input, top_k=5)
            augmented_docs = st.session_state["augmentation_module"].augment(multi_step_docs)
            re_ranked_docs = st.session_state["reranker"].rerank(user_input, augmented_docs, top_k=3)
            if all(isinstance(d, str) for d in re_ranked_docs):
                final_docs = re_ranked_docs
            else:
                final_docs = [d.page_content for d in re_ranked_docs]
            prompt = st.session_state["prompt_engineer"].build_prompt(
                user_query=user_input,
                conversation_context=conversation_context,
                relevant_docs=final_docs
            )
            final_answer = st.session_state["groq_llm"].invoke(prompt)
            # Append the new Q&A pair to the conversation history list
            st.session_state["conversation_history"].append({
                "user": user_input,
                "assistant": final_answer
            })
        else:
            st.warning("Please enter a valid question.")
    
    ###################################################
    # 8) UPDATE CONVERSATION HISTORY PLACEHOLDER (AFTER INPUT)
    ###################################################
    with history_placeholder.container():
        st.markdown("### Conversation History")
        if st.session_state["conversation_history"]:
            for i, turn in enumerate(st.session_state["conversation_history"], start=1):
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="chat-message user">
                            <strong>Q{i}:</strong> {turn['user']}
                        </div>
                        <div class="chat-message assistant">
                            <strong>A{i}:</strong> {turn['assistant']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No conversation yet. Ask a question below!")

if __name__ == "__main__":
    main()