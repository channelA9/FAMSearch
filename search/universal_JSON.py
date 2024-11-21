# chatbot.py
import os
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
from langchain_community.chat_models import ChatCohere
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cohere import ChatCohere

# Settings
MEMORY_TYPE = "embedding_section"  # embedding_file, embedding_sentence
RESPONSE_STRUCTURE = "loose"  # loose, directed
OUTPUT_FILE = 'test-section-document.csv'

EMBEDDINGS_FILE = "section_embeddings.json"
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = 'text-embedding-3-large'

SIMILARITY_CUTOFF = 0.3  # generally tried .3 for documents, .5 for sentence
MAX_SECTION_REFERENCES = 4 # if more than this amount of sections are deemed relevant, cap it to this amount.
MAX_RESPONSE_TOKENS = 500
CHAT_PROVIDER = 'google'  # openai google anthropic cohere
CHAT_MODEL = 'gemini-1.5-flash'  # gpt-4o gpt-4o-mini gemini-1.5-pro gemini-1.5-flash claude-3-5-sonnet-latest claude-3-5-haiku-latest command-r-plus command-r

# load_dotenv()

class EmbeddingService:
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        if provider == 'openai':
            self.client = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        else:
            raise ValueError("Unsupported embedding provider")

    def get_embedding(self, text):
        return self.client.embed_query(text)

class ChatModel:
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        if provider == 'openai':
            self.client = ChatOpenAI(model=model_name, max_tokens=MAX_RESPONSE_TOKENS)
        elif provider == 'google':
            self.client = ChatGoogleGenerativeAI(model=model_name, max_tokens=MAX_RESPONSE_TOKENS)
        elif provider == 'anthropic':
            self.client = ChatAnthropic(model=model_name, max_tokens=MAX_RESPONSE_TOKENS)
        elif provider == 'cohere':
            self.client = ChatCohere(model=model_name, max_tokens=MAX_RESPONSE_TOKENS)
        else:
            raise ValueError("Unsupported chat model provider")

    def generate_response(self, context, query):
        messages = []
        if RESPONSE_STRUCTURE == 'loose':
            messages = [
                {"role": "system", "content": "You are an assistant that will answer the user's question using relevant regulations based on the Faculty Administration Manual."},
                {"role": "user", "content": f"Using the following sections from the FAM: [{context}] Answer the question: '{query}'."}
            ]
        elif RESPONSE_STRUCTURE == 'directed':
            messages = [
                {"role": "system", "content": "You are an assistant that, in order to answer the user's question, will only respond with quoted sections verbatim from the Faculty Administration Manual."},
                {"role": "user", "content": f"Quoting the following sections from the FAM: [{context}] Answer the question: '{query}'."}
            ]
        return self.client.invoke(input=messages).content

class ChatBot:
    def __init__(self, embedding_service, chat_model):
        self.embedding_service = embedding_service
        self.chat_model = chat_model

    def load_embeddings(self):
        # Load embeddings from a JSON file instead of a CSV
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
            embeddings_data = json.load(file)
        return pd.DataFrame(embeddings_data)

    def cosine_similarity(self, vec_a, vec_b):
        vec_a, vec_b = np.array(vec_a), np.array(vec_b)
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def find_relevant_sections(self, df, query):
        query_embedding = self.embedding_service.get_embedding(query)
        df['similarity'] = df['embedding'].apply(lambda x: self.cosine_similarity(x, query_embedding))
        relevant_sections = df[df['similarity'] >= SIMILARITY_CUTOFF].sort_values(by='similarity', ascending=False)
        return relevant_sections[['regulation', 'content', 'similarity']]

    def generate_final_answer(self, query, sections):
        context = "\n\n".join([f"Section {row['regulation']}: {row['content']}" for _, row in sections.iterrows()])
        response = self.chat_model.generate_response(context, query)
        return response

    def start_chat(self, a=None):
        print("Welcome! How can I assist you today?")
        query = a if a else input("Your question: ")
        
        df = self.load_embeddings()
        relevant_sections = self.find_relevant_sections(df, query)

        if relevant_sections.empty:
            print(f"No sections found with similarity above {SIMILARITY_CUTOFF * 100}%.")
            with open(OUTPUT_FILE, "a") as f:
                f.write(f'\n"{query}",{RESPONSE_STRUCTURE},{MEMORY_TYPE},{CHAT_MODEL},{MAX_RESPONSE_TOKENS},"No sections found to answer this with similarity above {SIMILARITY_CUTOFF * 100}%.",{SIMILARITY_CUTOFF}')
            return
        
        print("Relevant sections found:")
        print(relevant_sections[['regulation', 'similarity']])
        
        selected_sections = relevant_sections
        if len(selected_sections) > MAX_SECTION_REFERENCES: selected_sections = relevant_sections[0:MAX_SECTION_REFERENCES]

        answer = self.generate_final_answer(query, selected_sections)
        print("Answer:", answer)

        with open(OUTPUT_FILE, "a") as f:
             f.write(f'\n"{query}",{RESPONSE_STRUCTURE},{MEMORY_TYPE},{CHAT_MODEL},{MAX_RESPONSE_TOKENS},"{answer.replace("\n", " ").replace('"', '""')}",{SIMILARITY_CUTOFF}')
        time.sleep(1)

        # Uncomment to allow repeated chats
        # self.start_chat()

# Initialize the chatbot with specific provider configurations
embedding_service = EmbeddingService(provider=EMBEDDING_PROVIDER, model_name=EMBEDDING_MODEL)
chat_model = ChatModel(provider=CHAT_PROVIDER, model_name=CHAT_MODEL)

bot = ChatBot(embedding_service, chat_model)

def cycle():
    bot.start_chat("How can I get a research grant?")
    bot.start_chat("Tell me the steps to give a faculty member the OSRCA Award.")
    bot.start_chat("What is the university stance on Academic Freedom?")
    bot.start_chat("What are the ethical standards faculty should observe?")
    bot.start_chat("How would a faculty member conduct a faculty exchange?")
    bot.start_chat("To what extent can faculty organize and bargain?")
    bot.start_chat("How can the Senate conduct a referendum?")
    bot.start_chat("What is in the faculty senate constitution?")
    bot.start_chat("What are the responsibilities of school heads?")
    bot.start_chat("An ancillary unit has disbanded. What are the administrative steps to take?")
    print("Cycle complete")

if __name__ == "__main__":
    # bot.start_chat()
    cycle()
