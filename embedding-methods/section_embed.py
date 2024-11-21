import os
import re
from dotenv import load_dotenv
import pandas as pd
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

# Function to parse the document into sections
def parse_document_by_sections(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Regex to identify sections based on numbering (e.g., "FAM 035.3")
    sections = re.split(r'(FAM \d{3}(?:\.\d{1,2})?)', content)
    parsed_sections = []
    
    # Combine section headers with their respective content
    for i in range(1, len(sections), 2):
        section_header = sections[i].strip()
        section_content = sections[i + 1].strip()
        parsed_sections.append({"header": section_header, "content": section_content})
    
    return parsed_sections

# Function to parse the document by FAM rule identifiers
def parse_document_by_fam_rules(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex to identify FAM rules on a separate line
    rule_pattern = r'^\s*(FAM \d{3}(?:\.\d{1,3}|/\d{3}\.\d{1,3}|/\d{3})?)\s*$'
    rule_splitter = re.compile(rule_pattern, re.MULTILINE)

    # Split content based on rule headers
    segments = rule_splitter.split(content)
    
    parsed_rules = []
    rule_string = []
    for i in range(1, len(segments), 2):
        rule_identifier = segments[i].strip()  # Capture rule identifier
        rule_content = segments[i + 1].strip()  # Capture following content
        parsed_rules.append({"rule": rule_identifier, "content": rule_content})
        rule_string.append(rule_identifier)
    print(rule_string)
    return parsed_rules


# Function to compute embeddings for parsed content
def compute_embeddings(parsed_data, parse_type, max_tokens=8192):
    embeddings = []
    for item in parsed_data:
        header = item.get("header", item.get("rule", ""))
        content = item["content"]

        # Split content into smaller chunks if it exceeds the max token limit
        chunks = split_into_chunks(content, max_tokens)
        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            embeddings.append({
                "regulation": f"{header} - Part {index + 1}",
                "content": chunk,
                "embedding": embedding
            })

    # Save embeddings to a JSON file for the parse type
    output_file = f"{parse_type}_embeddings.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(embeddings, file, indent=4)
    print(f"Embeddings saved to {output_file}")


def split_into_chunks(content, max_tokens):
    words = content.split()  # Approximate token count using words
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        # Estimate chunk size based on word count (assume 1 word ~1 token)
        if len(current_chunk) >= max_tokens - 2000:  # Safety margin of 100 tokens
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Append the last chunk if it has any remaining words
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Main function to process the document
def main():
    document_path = "corpus/FAMComplete.txt"

    # Parse and embed by sections
    # sections = parse_document_by_sections(document_path)
    # compute_embeddings(sections, "sections")
    
    # Parse and embed by FAM rules
    fam_rules = parse_document_by_fam_rules(document_path)
    
    compute_embeddings(fam_rules, "fam_rules")

if __name__ == "__main__":
    main()
