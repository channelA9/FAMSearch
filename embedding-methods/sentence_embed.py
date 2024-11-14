# embedding_agent.py
import os
import re
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

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

# Function to load sentences from all text files in the 'corpus' folder
def load_sentences_from_directory():
    sentences = []
    corpus_dir = "corpus"
    
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            regulation = filename.replace("FAM", "").replace("-", ".").replace(".txt", "")
            file_path = os.path.join(corpus_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                
                # Split content into sentences using a basic regex
                split_sentences = re.split(r'(?<=[.!?]) +', content)
                
                for sentence in split_sentences:
                    sentence = sentence.strip()
                    if sentence:  # Check if sentence is non-empty
                        sentences.append({"regulation": regulation, "content": sentence})
    
    return pd.DataFrame(sentences)

# Precompute embeddings for each sentence and save to CSV
def generate_and_save_embeddings():
    sentences_df = load_sentences_from_directory()
    
    # Generate embeddings for each sentence and store in a new column
    sentences_df['embedding'] = sentences_df['content'].apply(get_embedding)
    
    # Save to CSV for later use
    sentences_df.to_csv("sentence_embeddings.csv", index=False)

if __name__ == "__main__":
    generate_and_save_embeddings()
    print("Sentence embeddings generated and saved to 'sentence_embeddings.csv'")
