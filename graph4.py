import os
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import plotly.express as px
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.chat_models import ChatOpenAI

# API Keys
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyDA5GvuW7EluhCIheHB8IkG3uJ3_hcx9MM"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_csv_text(csv_docs):
    dfs = [pd.read_csv(csv) for csv in csv_docs]
    combined_df = pd.concat(dfs, ignore_index=True)
    column_name = combined_df.columns[0]
    value_counts = combined_df[column_name].value_counts()
    charts = generate_charts(value_counts, "CSV Data Distribution")
    return charts

def get_excel_text(excel_docs):
    dfs = []
    for excel in excel_docs:
        df = pd.read_excel(excel, sheet_name=None)
        for sheet_df in df.values():
            dfs.append(sheet_df)
    combined_df = pd.concat(dfs, ignore_index=True)
    column_name = combined_df.columns[0]
    value_counts = combined_df[column_name].value_counts()
    charts = generate_charts(value_counts, "Excel Data Distribution")
    return charts

def generate_charts(value_counts, title):
    charts = {
        'pie chart': px.pie(values=value_counts.values, names=value_counts.index, title=title),
        'bar chart': px.bar(value_counts.index, value_counts.values, title=f"{title} (Bar Chart)"),
        'line chart': px.line(value_counts.index, value_counts.values, title=f"{title} (Line Chart)"),
        'combo chart': px.line(value_counts.index, value_counts.values, title=f"{title} (Combo Chart)").add_scatter(x=value_counts.index, y=value_counts.values, mode='markers', name='Data Points'),
        'scatter plot': px.scatter(value_counts.index, value_counts.values, title=f"{title} (Scatter Plot)"),
        'histogram': px.histogram(value_counts.index, value_counts.values, title=f"{title} (Histogram)"),
        'bubble chart': px.scatter(value_counts.index, value_counts.values, size=value_counts.values, title=f"{title} (Bubble Chart)"),
    }
    return charts

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    faiss_index = FAISS.from_texts(text_chunks, embedding=embeddings)
    faiss_index.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "answer is not available in the context."\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="thenlper/gte-small"), allow_dangerous_deserialization=True)
    docs = embeddings.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="Welcome Vodafone")
    st.header("Welcome to Vodafone")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
        csv_docs = st.file_uploader("Upload your CSV Files", accept_multiple_files=True, type=['csv'])
        excel_docs = st.file_uploader("Upload your Excel Files", accept_multiple_files=True, type=['xls', 'xlsx'])

        if 'charts' not in st.session_state:
            st.session_state.charts = {}

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                if csv_docs:
                    st.session_state.charts = get_csv_text(csv_docs)
                if excel_docs:
                    st.session_state.charts = get_excel_text(excel_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    def add_message():
        user_question = st.session_state.user_question
        if user_question:
            if user_question.lower().startswith("show chart"):
                chart_name = user_question[len("show chart"):].strip().lower()
                if chart_name in st.session_state.charts:
                    st.write(f"**Selected Chart:** {chart_name.capitalize()}")
                    st.plotly_chart(st.session_state.charts[chart_name])
                else:
                    st.write("Chart not found. Available charts are:")
                    for chart in st.session_state.charts.keys():
                        st.write(f"- {chart.capitalize()}")
            else:
                answer = user_input(user_question)
                st.session_state.conversation.append({"question": user_question, "answer": answer})
            st.session_state.user_question = ""

    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_input("Ask a Question from the Files or Request a Chart (e.g., 'Show chart Pie Chart')", key="user_question", on_change=add_message)
    with col2:
        if st.button("Erase"):
            st.session_state.user_question = ""
            st.session_state.conversation = []

    for chat in st.session_state.conversation:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**AI:** {chat['answer']}")

if __name__ == "__main__":
    main()
