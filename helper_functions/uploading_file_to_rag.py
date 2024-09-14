from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from PyPDF2 import PdfReader
import os
import streamlit as st
import time
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Set up the index
index_name = 'surgical-assistant'

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PineconeVectorStore(index=index, embedding=embedder)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to upload a PDF to Pinecone in batches
def upload_pdf_to_pinecone(pdf_path):
    st.write("📄 Extracting text from the PDF file...")
    pdf_text = extract_text_from_pdf(pdf_path)
    st.write("✅ Text extraction completed.")
    
    # Split the text into chunks (512 characters per chunk)
    st.write("🔄 Splitting text into smaller chunks for embedding...")
    chunks = [pdf_text[i:i+512] for i in range(0, len(pdf_text), 512)]
    st.write(f"✅ Text has been split into {len(chunks)} chunks.")

    # Generate embeddings and prepare document objects
    st.write("🧠 Generating embeddings and preparing documents...")
    documents = []
    uuids = []
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = embedder.embed_documents([chunk])[0]
            # Create a Document object for each chunk
            document = Document(
                page_content=chunk,
                metadata={"source": pdf_path.name, "chunk_index": i}
            )
            documents.append(document)
            uuids.append(str(uuid4()))
            st.write(f"✅ Prepared document {i + 1}/{len(chunks)}")
        except Exception as e:
            st.write(f"❌ Error embedding chunk {i + 1}. Error: {e}")
            break  # Stop embedding if an error occurs

    if len(documents) == 0:
        st.write("⚠️ No documents generated. Exiting.")
        return

    # Upsert documents to Pinecone using the add_documents function
    st.write("🚀 Uploading documents to Pinecone...")
    try:
        vector_store.add_documents(documents=documents, ids=uuids)
        st.write("✅ Documents successfully uploaded to Pinecone.")
    except Exception as e:
        st.write(f"❌ Error uploading documents to Pinecone. Error: {e}")

# Streamlit app for user to provide PDF path
st.title("Pinecone PDF Uploader with HuggingFace Embeddings")

# File uploader for local file path
pdf_file_path = st.file_uploader('Upload PDF', type='pdf')
btn = st.button('Upload to Pinecone')

if pdf_file_path and btn:
    st.write(f"📂 Using file: {pdf_file_path}")
    
    # Upload the PDF file to Pinecone
    upload_pdf_to_pinecone(pdf_file_path)
