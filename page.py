import streamlit as st
import pandas as pd
from search.universal_JSON import ChatBot, EmbeddingService, ChatModel

# Streamlit settings
st.set_page_config(page_title="FAM Chatbot", layout="wide")
st.title("Faculty Administration Manual (FAM) Chatbot")

# Sidebar for relevant sections
st.sidebar.title("Relevant Sections")
st.sidebar.write("Relevant sections based on your query will appear here.")

# Model and provider selection in the sidebar
st.sidebar.subheader("Choose Models and Providers")

# Chat Provider and Model Selection
chat_provider = st.sidebar.selectbox(
    "Select Chat Provider",
    ["openai", "google", "anthropic", "cohere"]  # Adjust providers as needed
)

chat_model = st.sidebar.selectbox(
    "Select Chat Model",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gemini-1.5-pro", "gemini-1.5-flash", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "command-r-plus", "command-r" ]  # Add more models if needed
)

# Loose vs Directed Mode Selection
response_mode = st.sidebar.selectbox(
    "Select Response Mode",
    ["loose", "directed"]
)

# Initialize the chatbot with selected settings
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = 'text-embedding-3-large'
EMBEDDINGS_FILE = "section_embeddings.json"
MAX_SECTION_REFERENCES = 4

embedding_service = EmbeddingService(provider=EMBEDDING_PROVIDER, model_name=EMBEDDING_MODEL)
chat_model = ChatModel(provider=chat_provider, model_name=chat_model)
chatbot = ChatBot(embedding_service, chat_model)

# Load embeddings once to save time
@st.cache_data
def load_embeddings():
    return chatbot.load_embeddings()

embeddings_df = load_embeddings()

# Chat interface
query = st.text_input("Ask your question:", placeholder="Type your query here...")

# Main screen layout to show chatbot response
chatbot_response_box = st.empty()  # For displaying chatbot responses in the main screen
fam_section_box = st.empty()  # For displaying the FAM section content in a bottom box

if query:
    # Find relevant sections
    relevant_sections = chatbot.find_relevant_sections(embeddings_df, query).sort_values(by='similarity', ascending=False)

    # Display relevant sections in the sidebar with buttons
    if not relevant_sections.empty:
        st.sidebar.subheader("Relevant Sections:")

        # Loop through each relevant section
        for idx, row in relevant_sections.iterrows():
            # Assuming the relevancy score is in the column 'relevancy'
            relevancy_percentage = row['similarity'] * 100  # Adjust this if your relevancy score is already a percentage

            # Regular display for other sections
            st.sidebar.write(f"**{row['regulation']}** - {relevancy_percentage:.2f}%")
            
            # Optional: Display buttons to open the relevant sections
            # if st.sidebar.button(f"Open {row['regulation']}", key=idx):
            #     # Display the selected FAM section in a separate bottom box
            #     fam_section_box.subheader(f"Content for {row['regulation']}")
            #     fam_section_box.write(row['content'])
    else:
        st.sidebar.write("No relevant sections found.")


    # Generate and display chatbot response
    chatbot_response_box.subheader("Chatbot Response")
    if not relevant_sections.empty:
        # Limit sections to MAX_SECTION_REFERENCES
        selected_sections = relevant_sections.head(MAX_SECTION_REFERENCES)
        chat_response = chatbot.generate_final_answer(query, selected_sections)
        chatbot_response_box.write(chat_response)
    else:
        chatbot_response_box.write("No relevant sections found to generate a response.")
