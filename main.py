import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import hashlib
import uuid

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Neo4j graph
@st.cache_resource
def init_neo4j():
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

graph = init_neo4j()

# Initialize Google Gemini model
@st.cache_resource
def init_llm():
    return GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

llm = init_llm()

# Create a custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["question", "context", "user_info"],
    template="""
    You are an Ayurvedic expert AI assistant. Use the following context from the Ayurvedic knowledge graph and user information to answer the user's question. If you don't know the answer, say you don't know.

    Context: {context}
    User Info: {user_info}

    Human: {question}
    AI: """
)

# Initialize the GraphCypherQAChain
@st.cache_resource
def init_chain():
    return GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        cypher_prompt=custom_prompt
    )

chain = init_chain()

# ... (rest of the code remains the same)