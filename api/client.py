import requests
import streamlit as st
def get_openAI_response(input_text):
    response=requests.post("https://localhost:8000/essay/invoke",
    json={"input": {'topic':input_text}})
    return response.json()['output']['content']

def get_ollama_response(input_text):
    response=requests.post("https://localhost:8000/poem/invoke",
    json={"input": {'topic':input_text}})
    return response.json()['output']

#streamlit framework
st.title("Langchain Ollama Client")
input_text= st.text_input("Write an essay on ")
input_text_1= st.text_input("Write a poem on ")

if input_text:
    st.write(get_openAI_response(input_text))

if input_text_1:
    st.write(get_ollama_response(input_text_1))