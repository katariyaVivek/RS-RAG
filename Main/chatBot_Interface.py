import os
import pandas as pd
import streamlit as st
import openai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingestion import ingest
from retriever import SelfQueryRetriever

import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(CURRENT_DIR, "../data/main-data/synthetic-resumes.csv")
FAISS_PATH = os.path.join(CURRENT_DIR, "../vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

welcome_message = """
  #### Introduction 🚀

  The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. ⚡

  The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
  This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

  #### Getting started 🛠️

  1. To set up, please add your OpenAI's API key. 🔑 
  2. Type in a job description query. 💬

  Hint: The knowledge base of the LLM has been loaded with a pre-existing vectorstore of [resumes](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/main-data/synthetic-resumes.csv) to be used right away. 
  In addition, you may also find example job descriptions to test [here](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/supplementary-data/job_title_des.csv).

  Please make sure to check the sidebar for more useful information. 💡
"""

info_message = """
  # Information

  ### 1. What if I want to use my own resumes?

  If you want to load in your own resumes file, simply use the uploading button above. 
  Please make sure to have the following column names: `Resume` and `ID`. 

  Keep in mind that the indexing process can take **quite some time** to complete. ⌛

  ### 2. What if I want to set my own parameters?

  You can change the RAG mode and the GPT's model type using the sidebar options above. 

  About the other parameters such as the generator's *temperature* or retriever's *top-K*, I don't want to allow modifying them for the time being to avoid certain problems. 
  FYI, the temperature is currently set at `0.1` and the top-K is set at `5`.  

  ### 3. Is my uploaded data safe? 

  Your data is not being stored anyhow by the program. Everything is recorded in a Streamlit session state and will be removed once you refresh the app. 

  However, it must be mentioned that the **uploaded data will be processed directly by OpenAI's GPT**, which I do not have control over. 
  As such, it is highly recommended to use the default synthetic resumes provided by the program. 

  ### 4. How does the chatbot work? 

  The Chatbot works a bit differently to the original structure proposed in the paper so that it is more usable in practical use cases.

  For example, the system classifies the intent of every single user prompt to know whether it is appropriate to toggle RAG retrieval on/off. 
  The system also records the chat history and chooses to use it in certain cases, allowing users to ask follow-up questions or tasks on the retrieved resumes.
"""

about_message = """
  # About

  This small program is a prototype designed out of pure interest as additional work for the author's Bachelor's thesis project. 
  The aim of the project is to propose and prove the effectiveness of RAG-based models in resume screening, thus inspiring more research into this field.

  The program is very much a work in progress. I really appreciate any contribution or feedback on [GitHub]().

  If you are interested, please don't hesitate to give me a star. ⭐
"""


# Streamlit page configuration
st.set_page_config(page_title="Resume Screening GPT")
st.title("Resume Screening GPT")

# Initialize session state
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content=welcome_message)]
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(DATA_PATH)
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    if "rag_pipeline" not in st.session_state:
        vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
        st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
    if "resume_list" not in st.session_state:
        st.session_state.resume_list = []

initialize_session_state()

# Upload file handling
@st.cache(allow_output_mutation=True)
def upload_file(uploaded_file):
    if uploaded_file is not None:
        try:
            df_load = pd.read_csv(uploaded_file)
            if "Resume" not in df_load.columns or "ID" not in df_load.columns:
                raise ValueError("Please include the following columns in your data: 'Resume', 'ID'.")
            vectordb = ingest(df_load, "Resume", st.session_state.embedding_model)
            return SelfQueryRetriever(vectordb, df_load)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return None
    return st.session_state.rag_pipeline

# API key validation
@st.cache
def check_openai_api_key(api_key: str):
    openai.api_key = api_key
    try:
        openai.models.list()
        return True
    except openai.AuthenticationError:
        return False

# Main interaction
user_query = st.text_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    response = "Response to the user query would be generated here."
    st.session_state.chat_history.append(AIMessage(content=response))

# Sidebar configuration
with st.sidebar:
    api_key = st.text_input("OpenAI's API Key", type="password")
    if api_key and not check_openai_api_key(api_key):
        st.error("Invalid API key. Please enter a valid key.")
    uploaded_file = st.file_uploader("Upload resumes", type=["csv"])
    retriever = upload_file(uploaded_file)
    st.button("Clear conversation", on_click=lambda: st.session_state.chat_history.clear())

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        st.write(message.content)
    elif isinstance(message, HumanMessage):
        st.write(message.content)