import os
import pandas as pd
import json
import ast

from dotenv import load_dotenv
from openai import OpenAI

# Settings
MODEL = 'gpt-4o-mini'
MAX_RESPONSE_TOKENS = 300
OUTPUT_FILE = 'test-1.csv'

# Set up OpenAI client using API key from environment variable

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load documents from CSV
documents_df = pd.read_csv("./corpus/directory.csv")

# Convert the CSV content to a formatted string for GPT-4 to process
def get_csv_text():
    csv_text = ""
    for _, row in documents_df.iterrows():
        regulation = row['regulation']
        title = row['title']
        csv_text += f"{regulation}: {title}\n"
    return csv_text.strip()

# Function for the first query to identify relevant document sections
def find_relevant_sections(query):
    csv_content = get_csv_text()
    
    # Ask GPT-4 to pick relevant sections
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a bot that will identify relevant regulations from the Faculty Administration Manual (FAM) based on the user's question and return a JSON response."},
            {"role": "user", "content": f"The following is the Faculty Administration Manual:\n\n{csv_content}\n\nBased on this manual, identify the relevant section titles for the question: '{query}'. Please return the result as a JSON array with section numbers, like this: {{sections: ['35.3', '35.4', '35.6']}}"}
        ],
        max_tokens=100,
        temperature=0
    )
    
    # Access the content of the message correctly and clean up code block markers
    response_content = response.choices[0].message.content.strip()
    print(response_content)
    # Parse GPT-4's response to get the section numbers
    try:
        relevant_sections = ast.literal_eval(response_content)
        return relevant_sections['sections']
    except (ValueError, KeyError):
        print("Could not parse GPT-4's response.")
        return []


# Function for the second query to generate a final answer based on selected sections
def generate_final_answer(query, sections):
    # Concatenate the text of relevant sections
    detailed_text = ''

    for regulation in sections:
        parsedRegulation = regulation.split('.')
        chapter = parsedRegulation[0]
        section = parsedRegulation[1]
        fileDir = f'corpus/FAM{chapter}-{section}.txt'
        file = open(fileDir)
        detailed_text += file.read()
    
    # Send the relevant text to GPT-4 to get a final answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that will answer the user's question using relevant regulations based on the Faculty Administration Manual."},
            {"role": "user", "content": f"Using the following sections from the FAM:\n\n{detailed_text}\n\nAnswer the question: '{query}'."}
        ],
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=0
    )
    
    # Print the final answer
    answer = response.choices[0].message.content.strip()
    print("Answer:", answer)

# Main function to run the bot
def chat_bot():
    print("Welcome to FAMSearch! How can I assist you today?")
    query = input("Your question: ")

    # First cycle: Identify relevant sections
    sections = find_relevant_sections(query)
    if not sections:
        print("No relevant sections found. Please try rephrasing your question.")
        return
    
    print("Identified relevant sections:", sections)
    
    # Second cycle: Generate the detailed answer
    generate_final_answer(query, sections)

    f = open(OUTPUT_FILE, "a")
    f.write(f'\n"{query}",loose,embedding_sentence,{MODEL},{MAX_RESPONSE_TOKENS},"{answer.replace("\n", "")}",{SIMILARITY_CUTOFF}')
    f.close()

    chat_bot()

# Run the bot
if __name__ == "__main__":
    chat_bot()
