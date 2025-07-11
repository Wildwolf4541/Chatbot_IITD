# -*- coding: utf-8 -*-
"""chatbot with message aware.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1juE4Ue3CiJHhvmMjnmlZQ3Ek0f1wmRZj
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture --no-stderr
# %pip install --upgrade --quiet langchain langchain-community langchainhub langchain-chroma beautifulsoup4
# !pip install -q langchain_google_genai

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG_With_Memory"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCKvWXXEOqUzo3u8WxoHG99fAdyIE4ZHOU"

import warnings
warnings.filterwarnings('ignore')

from langchain_google_genai import GoogleGenerativeAIEmbeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",convert_system_message_to_human=True)

print(model.invoke("hi").content)

import bs4
from langchain import hub

from langchain.chains import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader

from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from bs4 import SoupStrainer
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load webpage content
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=SoupStrainer(class_=("post-content", "post-title","post-headers")))
)

# Load documents
doc = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)

# Set up Gemini embeddings (replace API key with your actual key safely)
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="AIzaSyCKvWXXEOqUzo3u8WxoHG99fAdyIE4ZHOU"
)

vectorstore= Chroma.from_documents(documents=splits,embedding=gemini_embeddings)
retriever=vectorstore.as_retriever()

retriever

system_prompt=(
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer the questions."
    "If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

chat_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]

)

question_answering_chain= create_stuff_documents_chain(model,chat_prompt)

rag_chain=create_retrieval_chain(retriever,question_answering_chain)

response=rag_chain.invoke({"input":"What is MRKL?"})

response["answer"]

from langchain.chains import create_history_aware_retriever

from langchain.chains import create_history_aware_retriever

retriever_prompt=(
    "Given a chat history and the latest user question which might referenece context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do not answer the question, just reformulate it if needed and otherwise return it as it is."
)

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever=create_history_aware_retriever(model,retriever,contextualize_q_prompt)

from langchain.chains import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer the user's question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=qa_prompt
)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.messages import HumanMessage, AIMessage

chat_history=[]

question1 = "What is Task Decomposition?"

message1 = rag_chain.invoke({"input": question1, "chat_history": chat_history})

message1["answer"]

chat_history.extend(
    [
        HumanMessage(content=question1),
        AIMessage(content=message1["answer"])
    ]
)

question2="What are the common ways of doing it?"
message2=rag_chain.invoke({"input":question2,"chat_history":chat_history})

print(message2["answer"])

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    }, # constructs a key "abc123" in `store`.
)["answer"]

store

conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]

for message in store["abc123"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"
    print(f"{prefix}: {message.content}\n")

