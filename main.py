import streamlit as st
import os
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
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

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    hashed_password = hash_password(password)
    user_id = str(uuid.uuid4())
    query = f"""
    CREATE (u:User {{id: '{user_id}', username: '{username}', password: '{hashed_password}'}})
    RETURN u.id as user_id
    """
    result = graph.query(query)
    return result[0]['user_id']

def authenticate_user(username, password):
    hashed_password = hash_password(password)
    query = f"""
    MATCH (u:User {{username: '{username}', password: '{hashed_password}'}})
    RETURN u.id as user_id
    """
    result = graph.query(query)
    return result[0]['user_id'] if result else None

def get_user_info(user_id):
    query = f"""
    MATCH (u:User {{id: '{user_id}'}})
    OPTIONAL MATCH (u)-[:HAS_SYMPTOM]->(s:Symptom)
    OPTIONAL MATCH (u)-[:HAS_CONDITION]->(c:Condition)
    OPTIONAL MATCH (u)-[:HAS_LIFESTYLE]->(l:Lifestyle)
    RETURN u.username as username, 
           collect(DISTINCT s.name) as symptoms, 
           collect(DISTINCT c.name) as conditions, 
           collect(DISTINCT l.name) as lifestyle
    """
    result = graph.query(query)
    return result[0] if result else None

def update_user_info(user_id, info_type, info_value):
    query = f"""
    MATCH (u:User {{id: '{user_id}'}})
    MERGE (i:{info_type} {{name: '{info_value}'}})
    MERGE (u)-[:HAS_{info_type.upper()}]->(i)
    """
    graph.query(query)

# Streamlit app
st.title("Personalized Ayurvedic Health Assistant")

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# User authentication
if not st.session_state.user_id:
    st.subheader("Login or Create Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    with col2:
        if st.button("Create Account"):
            if username and password:
                user_id = create_user(username, password)
                st.session_state.user_id = user_id
                st.success("Account created successfully!")
                st.experimental_rerun()
            else:
                st.error("Please enter a username and password")
else:
    user_info = get_user_info(st.session_state.user_id)
    st.sidebar.subheader(f"Welcome, {user_info['username']}!")

    # User profile update
    st.sidebar.subheader("Update Your Profile")
    info_type = st.sidebar.selectbox("Information Type", ["Symptom", "Condition", "Lifestyle"])
    info_value = st.sidebar.text_input(f"Add {info_type}")
    if st.sidebar.button("Add"):
        update_user_info(st.session_state.user_id, info_type, info_value)
        st.sidebar.success(f"{info_type} added successfully!")
        st.experimental_rerun()

    # Display user info
    st.sidebar.subheader("Your Information")
    st.sidebar.write(f"Symptoms: {', '.join(user_info['symptoms'])}")
    st.sidebar.write(f"Conditions: {', '.join(user_info['conditions'])}")
    st.sidebar.write(f"Lifestyle: {', '.join(user_info['lifestyle'])}")

    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.experimental_rerun()

    # Chat interface
    st.subheader("Chat with Ayurvedic Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about Ayurvedic treatments, symptoms, or conditions"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate AI response
        with st.chat_message("assistant"):
            user_info_str = f"Username: {user_info['username']}, Symptoms: {', '.join(user_info['symptoms'])}, Conditions: {', '.join(user_info['conditions'])}, Lifestyle: {', '.join(user_info['lifestyle'])}"
            response = chain({"question": prompt, "user_info": user_info_str})
            st.markdown(response['result'])

            # Optionally display intermediate steps (Cypher query and raw response)
            with st.expander("See explanation"):
                st.write("Cypher Query:")
                st.code(response['intermediate_steps'][0]['query'])
                st.write("Raw Response:")
                st.write(response['intermediate_steps'][0]['context'])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response['result']})

# Display information about the app
st.sidebar.title("About")
st.sidebar.info(
    "This Personalized Ayurvedic Health Assistant uses a knowledge graph to provide "
    "tailored information about Ayurvedic treatments, symptoms, and conditions. "
    "Create an account or log in to get personalized recommendations based on your profile."
)