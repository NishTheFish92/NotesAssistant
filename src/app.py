import streamlit as st
from langchain.vectorstores import FAISS
from utils import init_embeddings, init_llm
from create_chains import chainer


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    html, body, .stApp {
        background: linear-gradient(135deg, #181818 0%, #232323 60%, #b36b00 100%) !important;
        font-family: 'Montserrat', sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #232323;
        color: #f5f5f5;
        border: 1.5px solid #ffb366;
        border-radius: 8px;
        font-size: 1.1rem;
        padding: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ffb366 0%, #ff8c42 100%);
        color: #232323;
        border-radius: 8px;
        border: none;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 10px 24px;
        margin-top: 12px;
        box-shadow: 0 2px 8px rgba(255, 179, 102, 0.2);
        transition: background 0.3s;
    }
    

    h1 {
        color: #ffb366 !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        letter-spacing: 2px;
        margin-top: 32px;
        text-align: center;
        text-shadow: 0 2px 8px rgba(255, 179, 102, 0.2);
    }
    label {
        color: #ffb366 !important;
        font-size: 1.08rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)



st.markdown("<h1>NotesAssistant</h1>", unsafe_allow_html=True)


with st.container():
    show_card = True
    response = None
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("<span style='color:#ffb366;font-size:1.08rem;font-weight:500;'>Ask your question below:</span>", unsafe_allow_html=True)
    query = st.text_input("Your question", "", key="query_input")

    @st.cache_resource(show_spinner=False)
    def get_chain():
        embeddings = init_embeddings()
        llm = init_llm()
        vectorstore = FAISS.load_local("faiss_index", embeddings)
        return chainer(vectorstore, llm)

    retrieval_chain = get_chain()

    if st.button("Get Answer", key="get_answer_btn") and query:
        response = retrieval_chain.invoke({"input": query})
        st.markdown(response["answer"], unsafe_allow_html=True)

 
    if query or response:
        st.markdown('</div>', unsafe_allow_html=True)
