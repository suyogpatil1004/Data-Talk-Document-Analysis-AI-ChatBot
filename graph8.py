import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import plotly.express as px

# Configure Google Generative AI
GOOGLE_API_KEY =
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_csv_text(csv_docs, column_name):
    dfs = [pd.read_csv(csv) for csv in csv_docs]
    combined_df = pd.concat(dfs, ignore_index=True)
    if column_name in combined_df.columns:
        value_counts = combined_df[column_name].value_counts()
        charts = generate_charts(value_counts, f"CSV Data Distribution for {column_name}")
    else:
        charts = {"error": f"Column '{column_name}' not found in the CSV files."}
    return charts

def get_excel_text(excel_docs, column_name):
    dfs = []
    for excel in excel_docs:
        df = pd.read_excel(excel, sheet_name=None)
        for sheet_df in df.values():
            dfs.append(sheet_df)
    combined_df = pd.concat(dfs, ignore_index=True)
    if column_name in combined_df.columns:
        value_counts = combined_df[column_name].value_counts()
        charts = generate_charts(value_counts, f"Excel Data Distribution for {column_name}")
    else:
        charts = {"error": f"Column '{column_name}' not found in the Excel files."}
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
        # Funnel chart requires Plotly Express >= 5.0.0
        'funnel chart': px.funnel(value_counts.index, value_counts.values, title=f"{title} (Funnel Chart)"),
        'box plot': px.box(value_counts.index, value_counts.values, title=f"{title} (Box Plot)")
    }
    return charts

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
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "answer is not available in the context."\n\n
    Context:\n{context}\n
    Question:\n{question}\n
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
    st.set_page_config(page_title="Welcome Vodafone")
    st.header("Welcome to Vodafone")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
        csv_docs = st.file_uploader("Upload your CSV Files", accept_multiple_files=True, type=['csv'])
        excel_docs = st.file_uploader("Upload your Excel Files", accept_multiple_files=True, type=['xls', 'xlsx'])

        column_name = st.text_input("Enter the column name for chart generation:")

        if 'charts' not in st.session_state:
            st.session_state.charts = {}

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                if csv_docs:
                    st.session_state.charts = get_csv_text(csv_docs, column_name)
                if excel_docs:
                    st.session_state.charts = get_excel_text(excel_docs, column_name)
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

    st.text_input("Ask a Question from the Files or Request a Chart (e.g., 'Show chart Pie Chart')", key="user_question", on_change=add_message)

    for chat in st.session_state.conversation:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**AI:** {chat['answer']}")

if __name__ == "__main__":
    main()
