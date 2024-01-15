# Loading necessary libraries.

import pinecone
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']
model = SentenceTransformer("all-MiniLM-L6-v2")
# Initialize pinecone
pinecone.init(
    api_key = '5e348161-8c15-4446-b5c2-ce21a1f7db10',
    environment = 'gcp-starter'
)
index = pinecone.Index('chatbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    # Generating messages for system,user and assistant.
    messages = [
        {"role":"system", "content":" Wonderful assistant"},
        {"role":"user", "content":conversation},
        {"role":"assistant","content":query}
    ]

    response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    #prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].message.content

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

