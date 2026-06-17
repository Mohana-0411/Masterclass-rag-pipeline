import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Streamlit Page Configuration
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="💼", layout="centered")
st.title("💼 Zyro Dynamics HR Help Desk")
st.markdown("Welcome to your internal employee portal. Ask any question regarding company policies, benefits, leaves, or payroll.")

# 2. Caching the RAG Components to prevent rebuilding on every rerun
@st.cache_resource(show_spinner="Initializing HR Database & Embeddings...")
def initialize_rag_pipeline():
    corpus_path = "/kaggle/input/competitions/niat-masterclass-rag-challenge/zyro-dynamics-hr-corpus/"
    # Fallback to local directory if deployed on Streamlit cloud
    if not os.path.exists(corpus_path):
        corpus_path = "./zyro-dynamics-hr-corpus/" 
        
    # Load Documents
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    # Chunk Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector store & Retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

@st.cache_resource
def get_llm():
    # Detect available API keys and pull appropriate model dynamically
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
    elif os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_output_tokens=512)
    elif os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=512)
    else:
        st.error("Missing LLM Provider API Key. Please configure your secrets/environment variables.")
        st.stop()

# Initialize background assets
try:
    retriever = initialize_rag_pipeline()
    llm = get_llm()
except Exception as e:
    st.error(f"Error setup pipeline: {e}")
    st.stop()

# 3. Prompts & Core Logic Definition
OOS_PROMPT = ChatPromptTemplate.from_template("""
You are an intent classification security guardrail for an internal corporate HR help desk.
Analyze the user's input and determine if it is a legitimate Human Resources (HR) question 
(e.g., benefits, leaves, payroll, code of conduct, onboarding, company policies).

If it is related to company HR policies or workplace questions, respond with exactly: "HR_RELATED"
If it is unrelated to HR, respond with exactly: "OUT_OF_SCOPE"

User input: {question}
Classification:""")

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful, precise, and professional HR assistant for Zyro Dynamics. 
Answer the employee's question based strictly on the provided policy context. 
If the context does not contain the answer, politely state that you do not know.

Context:
{context}

Question: {question}
Answer:""")

REFUSAL_MESSAGE = "I am an internal HR assistant for Zyro Dynamics. I can only assist you with company policy, benefits, payroll, leaves, or workplace-related inquiries."

def ask_bot(question: str):
    # Guardrail Check
    guard_chain = OOS_PROMPT | llm | StrOutputParser()
    classification = guard_chain.invoke({"question": question}).strip().upper()
    
    if "OUT_OF_SCOPE" in classification:
        return {"answer": REFUSAL_MESSAGE, "source_documents": []}
    
    # RAG Execution
    retrieved_docs = retriever.invoke(question)
    context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": question})
    
    return {"answer": answer, "source_documents": retrieved_docs}

# 4. Streamlit Chat Interface Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Viewed References"):
                for src in msg["sources"]:
                    st.caption(src)

# Handle new user input
if user_query := st.chat_input("Ask a question about company policy..."):
    # Render user query
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Reviewing company policies..."):
            res = ask_bot(user_query)
            answer = res["answer"]
            docs = res["source_documents"]
            
            # Format source strings
            sources_list = []
            if docs:
                for doc in docs:
                    name = os.path.basename(doc.metadata.get('source', 'Policy_Document.pdf'))
                    page = doc.metadata.get('page', 0) + 1
                    sources_list.append(f"📄 {name} — Page {page}")
            
            # Print response
            st.markdown(answer)
            if sources_list:
                with st.expander("📚 Viewed References"):
                    for src in sources_list:
                        st.caption(src)
                        
    # Save response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources_list if sources_list else None
    })