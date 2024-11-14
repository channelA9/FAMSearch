# embedding_agent.py
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from openai import OpenAI
import json

# Set up OpenAI client using API key from environment variable
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

# Function to embed text using OpenAI's embedding API
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# Function to load documents from the 'corpus' folder
def load_documents_from_directory():
    documents = []
    corpus_dir = "corpus"
    
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            regulation = filename.replace("FAM", "").replace("-", ".").replace(".txt", "")
            file_path = os.path.join(corpus_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
            
            documents.append({"regulation": regulation, "content": content})
    
    return pd.DataFrame(documents)

# Precompute embeddings for each document and save to CSV
def generate_and_save_embeddings():
    documents_df = load_documents_from_directory()
    
    # Generate embeddings and store in a new column
    documents_df['embedding'] = documents_df['content'].apply(get_embedding)
    
    # Save to CSV for later use
    documents_df.to_csv("document_embeddings.csv", index=False)

if __name__ == "__main__":
    generate_and_save_embeddings()
    print("Embeddings generated and saved to 'document_embeddings.csv'")
