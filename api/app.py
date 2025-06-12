from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server for Langchain",
)

# Gemini chat model
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")

# Ollama llama2 model
llama_model = Ollama(model="llama2")

# Register Gemini chat model route
add_routes(app, gemini_model, path="/gemini_chat")

# Prompts
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words.")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words.")

# Chains for essay and poem using llama2
essay_chain = LLMChain(prompt=prompt1, llm=llama_model)
poem_chain = LLMChain(prompt=prompt2, llm=llama_model)

# Register routes for essay and poem
add_routes(app, essay_chain, path="/essay")
add_routes(app, poem_chain, path="/poem")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
