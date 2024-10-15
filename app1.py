import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Set the API key
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_csv_text(csv_docs):
    text = ""
    for csv in csv_docs:
        df = pd.read_csv(csv)
        text += df.to_string(index=False)
    return text

def get_excel_text(excel_docs):
    text = ""
    for excel in excel_docs:
        df = pd.read_excel(excel, sheet_name=None)
        for sheet_name, sheet_df in df.items():
            text += f"Sheet: {sheet_name}\n"
            text += sheet_df.to_string(index=False)
            text += "\n\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="Welcome to Words Universe")
    st.header("Welcome to Words Universe")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
        csv_docs = st.file_uploader("Upload your CSV Files", accept_multiple_files=True, type=['csv'])
        excel_docs = st.file_uploader("Upload your Excel Files", accept_multiple_files=True, type=['xls', 'xlsx'])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                if csv_docs:
                    raw_text += get_csv_text(csv_docs)
                if excel_docs:
                    raw_text += get_excel_text(excel_docs)
                
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    def add_message():
        user_question = st.session_state.user_question
        if user_question:
            answer = user_input(user_question)
            st.session_state.conversation.append({"question": user_question, "answer": answer})
            st.session_state.user_question = ""

    st.text_input("Ask a Question from the Files", key="user_question", on_change=add_message)

    for chat in st.session_state.conversation:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**AI:** {chat['answer']}")

if __name__ == "__main__":
    main()
