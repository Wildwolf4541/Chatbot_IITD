import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable

# Your LangChain setup (replace with your actual objects)
from chatbot_with_message_aware import history_aware_retriever, question_answer_chain
from langchain.chains import create_retrieval_chain

# Create the RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 RAG Chatbot with History Awareness")

# User input
user_input = st.chat_input("Ask a question")

# Display past conversation
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle new user input
if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Invoke the RAG chain
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response["answer"])

    # Add assistant response to history
    st.session_state.chat_history.append(AIMessage(content=response["answer"]))
