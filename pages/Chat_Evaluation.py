import streamlit as st
from utils import create_sidebar
from trulens_eval import TruChain, Feedback, Huggingface, Tru, feedback
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from langchain.chains import RetrievalQA
from trulens.apps.custom import TruCustomApp

create_sidebar()

tru = Tru()

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

# OpenAI as feedback provider
llm = feedback.OpenAI()

# Question/answer relevance between overall question and answer.
qa_relevance = Feedback(llm.relevance).on_input_output()

api_key = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)
assistant = pc.assistant.Assistant(
    assistant_name="pineapple-employee-assistant-bot", 
)

# wrap with TruLens
truchain = TruCustomApp(assistant,
    app_id='Chat_QA101',
    app_name="Chat_QA101 Assistant RAG",
    feedbacks=[f_lang_match, qa_relevance])

assistant.chat("What should be done for walk-in guest?")

tru.run_dashboard()
