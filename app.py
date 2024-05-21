from flask import Flask, request, jsonify
from langchain_core.prompts import PromptTemplate
import http.client
import json
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import logging
import pickle
import numpy as np
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Retrieve RapidAPI key from environment variable
rapidapi_key = os.getenv("RAPIDAPI_KEY")
if not rapidapi_key:
    raise ValueError("RapidAPI key not found. Please set it in the .env file.")

# Load pre-computed embeddings and texts
with open('embeddings.pkl', 'rb') as f:
    embeddings, texts = pickle.load(f)

# Initialize sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Create a text-to-index mapping
text_mapping = {text: i for i, text in enumerate(texts)}

# Function for similarity search
def retrieve_info(query):
    logging.debug(f"Query: {query}")
    query_embedding = model.encode([query], convert_to_tensor=False)
    logging.debug(f"Query embedding shape: {query_embedding.shape}")
    distances, indices = index.search(query_embedding, k=3)
    logging.debug(f"Distances: {distances}")
    logging.debug(f"indices: {indices}")
    similar_texts = [texts[index] for index in indices[0]]
    return similar_texts

# Setup PromptTemplate
template = """
You are a world-class business development representative. 
I will share a prospect's message with you, and you will give me the best answer that 
I should send to this prospect based on past best practices, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practices, 
in terms of length, tone of voice, logical arguments, and other details.

2/ If the best practice is irrelevant, then try to mimic the style of the best practice to the prospect's message.

Below is a message I received from the prospect:
{message}

Here is a list of best practices of how we normally respond to prospects in similar scenarios:
{best_practice}

Please write a detailed and informative response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

# Function to interact with the new API
def generate_response(message):
    best_practice = retrieve_info(message)
    formatted_prompt = prompt.format(message=message, best_practice="\n".join(best_practice))
    
    conn = http.client.HTTPSConnection("chatgpt-42.p.rapidapi.com")
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        "web_access": False
    })

    headers = {
        'content-type': "application/json",
        'X-RapidAPI-Key': rapidapi_key,
        'X-RapidAPI-Host': "chatgpt-42.p.rapidapi.com"
    }

    conn.request("POST", "/gpt4", payload, headers)
    res = conn.getresponse()
    data = res.read()
    response_data = json.loads(data.decode("utf-8"))

    logging.debug(f"API Response: {response_data}")  # Log the full response

    if 'result' in response_data:
        return response_data["result"]
    else:
        logging.error(f"Unexpected API response format: {response_data}")
        return "Error: Unexpected API response format."

# Define Flask routes
@app.route('/generate_response', methods=['POST'])
def generate_response_route():
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = generate_response(message)
    return jsonify({"response": response})

