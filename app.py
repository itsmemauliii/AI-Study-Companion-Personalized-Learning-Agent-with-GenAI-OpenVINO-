import streamlit as st
from utils import load_model, generate_response

st.set_page_config(page_title="AI Study Companion", page_icon=":books:")

st.title("AI Study Companion")
st.write("Personalized Learning Agent powered by GenAI & Intel® OpenVINO™")

# Load model (optimized with OpenVINO)
model, tokenizer = load_model()

st.sidebar.header("Your Study Preferences")
topic = st.sidebar.text_input("Topic you want to learn about", "Machine Learning")
difficulty = st.sidebar.selectbox("Select difficulty level", ["Beginner", "Intermediate", "Advanced"])

if st.sidebar.button("Generate Study Plan"):
    with st.spinner("Generating personalized study plan..."):
        prompt = f"Create a {difficulty} level study plan on {topic}."
        plan = generate_response(model, tokenizer, prompt)
        st.subheader("Your Personalized Study Plan")
        st.write(plan)

st.header("Ask me anything!")
question = st.text_area("Enter your question")

if st.button("Get Answer"):
    if question.strip() != "":
        with st.spinner("Getting answer..."):
            answer = generate_response(model, tokenizer, question)
            st.subheader("Answer")
            st.write(answer)
    else:
        st.warning("Please enter a question.")
