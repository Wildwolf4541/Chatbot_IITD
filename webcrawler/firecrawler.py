import streamlit as st
import os
import tempfile
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import deque
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables and set API keys
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit page configuration
st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 MY CHATBOT")
st.markdown("*Ask questions about web content, PDF documents, or have general conversations*")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "general_chain" not in st.session_state:
    st.session_state.general_chain = None
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = []

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        convert_system_message_to_human=True,
        temperature=0.7
    )

llm = get_llm()

# Initialize general conversation chain
@st.cache_resource
def get_general_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the user queries in a friendly and informative manner."),
        ("user", "Question: {question}")
    ])
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

if st.session_state.general_chain is None:
    st.session_state.general_chain = get_general_chain()

def get_internal_links(url, soup, base_domain):
    """Extract internal links from a webpage"""
    links = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Convert relative links to absolute
        absolute_url = urljoin(url, href)
        parsed_url = urlparse(absolute_url)
        
        # Check if it's an internal link (same domain)
        if parsed_url.netloc == base_domain or parsed_url.netloc == '':
            # Remove fragments and query parameters for deduplication
            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            if clean_url.endswith('/'):
                clean_url = clean_url[:-1]
            links.add(clean_url)
    
    return links

def bfs_crawl_website(start_url, max_pages=10, max_depth=3):
    """Crawl website using BFS to discover internal links"""
    try:
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc
        
        visited = set()
        queue = deque([(start_url, 0)])  # (url, depth)
        crawled_urls = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        while queue and len(crawled_urls) < max_pages:
            current_url, depth = queue.popleft()
            
            if current_url in visited or depth > max_depth:
                continue
                
            try:
                response = requests.get(current_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Only process HTML content
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                visited.add(current_url)
                crawled_urls.append(current_url)
                
                # Find internal links only if we haven't reached max depth
                if depth < max_depth:
                    internal_links = get_internal_links(current_url, soup, base_domain)
                    for link in internal_links:
                        if link not in visited and link != current_url:
                            queue.append((link, depth + 1))
                
            except Exception as e:
                st.warning(f"Failed to crawl {current_url}: {str(e)}")
                continue
        
        return crawled_urls
    
    except Exception as e:
        st.error(f"Error during BFS crawling: {str(e)}")
        return [start_url]  # Return original URL as fallback

def process_documents(documents, source_info):
    """Process documents and create/update vector store"""
    try:
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create or update vector store
        if st.session_state.vectorstore is None:
            vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        else:
            # Add new documents to existing vector store
            st.session_state.vectorstore.add_documents(splits)
            vectorstore = st.session_state.vectorstore
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create history-aware retriever
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Create QA chain
        qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Keep the answer concise.

{context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Store in session state
        st.session_state.retriever = retriever
        st.session_state.vectorstore = vectorstore
        st.session_state.rag_chain = rag_chain
        st.session_state.processed_sources.append(source_info)
        
        return len(splits)
        
    except Exception as e:
        raise e

# Sidebar for content loading
with st.sidebar:
    st.header("📄 Content Loader")
    
    # Web URL section
    st.subheader("🌐 Web Content")
    url = st.text_input("Enter URL to crawl:", placeholder="https://example.com")
    
    # BFS Crawler options
    with st.expander("🕷️ BFS Crawler Settings"):
        max_pages = st.slider("Max pages to crawl:", min_value=1, max_value=50, value=10)
        max_depth = st.slider("Max crawl depth:", min_value=1, max_value=5, value=3)
        enable_bfs = st.checkbox("Enable BFS crawling (crawl internal links)", value=True)
    
    if st.button("🔄 Fetch Web Content", use_container_width=True):
        if not url:
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Loading and embedding web content..."):
                try:
                    if enable_bfs:
                        # Use BFS crawler to get multiple URLs
                        crawled_urls = bfs_crawl_website(url, max_pages, max_depth)
                        st.info(f"🕷️ BFS Crawler found {len(crawled_urls)} pages")
                        
                        # Display crawled URLs
                        with st.expander("📋 Crawled URLs"):
                            for i, crawled_url in enumerate(crawled_urls, 1):
                                st.text(f"{i}. {crawled_url}")
                        
                        # Load all crawled URLs
                        loader = WebBaseLoader(web_paths=crawled_urls)
                        documents = loader.load()
                        
                        source_info = {
                            "type": "web_bfs", 
                            "source": url, 
                            "crawled_count": len(crawled_urls),
                            "crawled_urls": crawled_urls
                        }
                    else:
                        # Load single URL
                        loader = WebBaseLoader(web_paths=[url])
                        documents = loader.load()
                        source_info = {"type": "web", "source": url}
                    
                    if not documents:
                        st.error("No content found at the provided URL(s).")
                    else:
                        num_chunks = process_documents(documents, source_info)
                        if enable_bfs:
                            st.success(f"✅ Successfully processed {num_chunks} chunks from {len(crawled_urls)} web pages!")
                        else:
                            st.success(f"✅ Successfully processed {num_chunks} chunks from web content!")
                        
                except Exception as e:
                    st.error(f"❌ Failed to process web content: {str(e)}")
    
    # PDF upload section
    st.subheader("📋 PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files:",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select one or more PDF files to process"
    )
    
    if st.button("📚 Process PDF Files", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            total_chunks = 0
            processed_files = []
            
            with st.spinner("Processing PDF files..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded file to temporary directory
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_file_path = tmp_file.name
                        
                        # Load PDF content
                        loader = PyPDFLoader(tmp_file_path)
                        documents = loader.load()
                        
                        if documents:
                            num_chunks = process_documents(
                                documents, 
                                {"type": "pdf", "source": uploaded_file.name}
                            )
                            total_chunks += num_chunks
                            processed_files.append(uploaded_file.name)
                        
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
                    except Exception as e:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                        continue
            
            if processed_files:
                st.success(f"✅ Successfully processed {total_chunks} chunks from {len(processed_files)} PDF file(s)!")
                for filename in processed_files:
                    st.info(f"📄 {filename}")
    
    # Clear all content button
    if st.button("🗑️ Clear All Content", use_container_width=True):
        st.session_state.retriever = None
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.processed_sources = []
        st.success("✅ All content cleared!")
        st.rerun()
    
    # Clear chat history button
    if st.button("💬 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Display current status
    st.header("📊 Status")
    if st.session_state.retriever:
        st.success("✅ RAG System Ready")
        st.info("🤖 Using Gemini 1.5 Flash")
        
        # Show processed sources
        if st.session_state.processed_sources:
            st.subheader("📚 Loaded Sources:")
            for i, source in enumerate(st.session_state.processed_sources, 1):
                if source["type"] == "web":
                    st.markdown(f"🌐 **Web {i}:** {source['source']}")
                elif source["type"] == "web_bfs":
                    st.markdown(f"🕷️ **BFS Web {i}:** {source['source']} ({source['crawled_count']} pages)")
                elif source["type"] == "pdf":
                    st.markdown(f"📄 **PDF {i}:** {source['source']}")
        
        st.info("Ask questions about the loaded content!")
    else:
        st.info("💡 Load web content or PDF files to enable RAG mode")
        st.info("🤖 Using Gemini 1.5 Flash")
        st.info("Or ask general questions!")

# Main chat interface
st.header("💬 Chat Interface")

# Display chat history
for i, msg in enumerate(st.session_state.chat_history):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input
user_input = st.chat_input("Ask a question about the content or anything else...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.rag_chain and st.session_state.retriever:
                    # Use RAG chain if available
                    response = st.session_state.rag_chain.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history[:-1]  # Exclude the current message
                    })
                    answer = response.get("answer", "I couldn't generate a response.")
                    
                    # Show sources if available
                    if "context" in response and response["context"]:
                        with st.expander("📚 Sources Used"):
                            for i, doc in enumerate(response["context"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                                # Show metadata if available
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    if 'source' in doc.metadata:
                                        st.markdown(f"*From: {doc.metadata['source']}*")
                else:
                    # Use general conversation chain
                    answer = st.session_state.general_chain.invoke({"question": user_input})
                
                st.markdown(answer)
                
            except Exception as e:
                answer = f"❌ Error generating response: {str(e)}"
                st.error(answer)
    
    # Add assistant response to history
    st.session_state.chat_history.append(AIMessage(content=answer))

# # Footer
# st.markdown("---")
# st.markdown("*Built with Streamlit, LangChain, and Google Gemini 1.5 Flash*")

# # Installation instructions
# with st.expander("📦 Installation Requirements"):
#     st.code("""
# # Install required packages:
# pip install streamlit
# pip install langchain
# pip install langchain-google-genai
# pip install langchain-community
# pip install langchain-chroma
# pip install python-dotenv
# pip install pypdf
# pip install beautifulsoup4
# pip install lxml
# pip install requests
#     """, language="bash")