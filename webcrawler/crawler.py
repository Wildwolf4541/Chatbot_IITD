import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Web Crawler with Gemini", layout="centered")
st.title("🌐 LangChain Web Crawler with Gemini")
url = st.text_input("https://indianexpress.com/", placeholder="https://example.com")

if url:
    with st.spinner("🔍 Crawling and processing content..."):

        async def crawl_and_prepare(target_url):
            loader = AsyncChromiumLoader([target_url])
            docs = await loader.aload()

            html2text = Html2TextTransformer()
            docs_text = html2text.transform_documents(docs)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs_text)

            return chunks

        chunks = asyncio.run(crawl_and_prepare(url))

        # Gemini Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = FAISS.from_documents(chunks, embeddings)

        retriever = vectordb.as_retriever()

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ Done! Ask a question below.")

        query = st.text_input("Ask a question about the page", placeholder="What is this website about?")
        if query:
            with st.spinner("💡 Thinking..."):
                response = qa_chain.run(query)
                st.write("**Answer:**", response)
