import streamlit as st
from chat_server import answer_question

st.set_page_config(page_title="Documentation Chatbot", page_icon="🤖")

st.title("Documentation Chatbot 🤖")
st.write("Ask questions about your documentation website.")

# Session state to keep conversation
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("Your question:")

if user_input:
    # Get answer from chat_server.py
    response = answer_question(user_input)
    
    # Store in history
    st.session_state.history.append({"user": user_input, "bot": response})
    
# Display conversation history
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")
