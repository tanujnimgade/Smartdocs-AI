import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import tempfile
import os

MODEL_ID = os.getenv("HF_MODEL_ID", "distilgpt2")

st.set_page_config(page_title="SmartDocs AI", layout="wide")
st.title("📄 SmartDocs AI - Multi PDF RAG Chatbot")

# ---------- CACHE HEAVY OBJECTS ----------

@st.cache_resource
def load_llm():
    pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    max_new_tokens=512
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------- FILE UPLOAD ----------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------- PROCESS PDFs ----------

if uploaded_files:

    st.success("Files uploaded successfully!")

    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            documents.extend(loader.load())
        os.unlink(tmp_file.name)

    if len(documents) == 0:
        st.error("No text extracted from PDFs")
        st.stop()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    texts = splitter.split_documents(documents)

    # Load cached components
    embeddings = load_embeddings()
    llm = load_llm()

    # Vector DB
    vectorstore = FAISS.from_documents(texts, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # ---------- QUESTION UI ----------

    st.subheader("💬 Ask Questions About Your Documents")

    query = st.text_input("Enter your question")

    if st.button("Ask"):

        if not query:
            st.warning("Please enter a question")
        else:
            with st.spinner("Thinking..."):

                try:
                    result = qa.invoke({"query": query})
                except:
                    result = qa(query)

            st.success("Answer generated")

            st.subheader("📌 Answer")
            st.write(result["result"])

            st.subheader("📄 Sources")
            for doc in result["source_documents"]:
                st.write(doc.metadata.get("source", "Unknown"))

else:
    st.info("Please upload at least one PDF to start.")
