import streamlit as st
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

# Force CPU and disable multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_default_device("cpu")

# -------------------------------
# Hugging Face API Key from Streamlit Secrets
# -------------------------------
HF_API_KEY = st.secrets.get("HF_API_KEY", "No key found")
st.write(f"HF_API_KEY: {HF_API_KEY[:6]}...")

# Cache text-generation pipeline
@st.cache_resource
def load_generator():
    return pipeline(
        "text-generation",
        model="distilgpt2",
        token=HF_API_KEY,
    )
generator = load_generator()

# -------------------------------
# Mock Project Data
# -------------------------------
projects = [
    {
        "id": 1,
        "name": "HiStudy Website",
        "client": "ABC Edu",
        "tasks": [
            {"task": "Design Landing Page", "status": "Done", "complexity": 3},
            {"task": "Integrate Payment Gateway", "status": "Pending", "complexity": 5},
            {"task": "Setup Language Settings", "status": "Pending", "complexity": 2},
        ],
        "deadline": "2025-10-20"
    },
    {
        "id": 2,
        "name": "Aiwave AI Tool",
        "client": "XYZ AI",
        "tasks": [
            {"task": "Build API", "status": "In Progress", "complexity": 4},
            {"task": "Add Dashboard", "status": "Pending", "complexity": 3},
        ],
        "deadline": "2025-11-05"
    }
]

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Rainbow Assistant", layout="wide")
st.title("🌈 Rainbow Assistant — AI Project Manager & Knowledge Base")

tabs = st.tabs(["🏠 Dashboard", "🤖 Project Assistant", "📅 Timeline Generator", "📚 Knowledge Base"])

# -------------------------------
# 1️⃣ DASHBOARD
# -------------------------------
with tabs[0]:
    st.header("Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Projects", len(projects))
    pending_tasks = sum([len([t for t in p["tasks"] if t["status"] != "Done"]) for p in projects])
    col2.metric("Pending Tasks", pending_tasks)
    upcoming_deadline = min([datetime.strptime(p["deadline"], "%Y-%m-%d") for p in projects])
    col3.metric("Next Deadline", upcoming_deadline.strftime("%Y-%m-%d"))
    st.write("Use the tabs to explore Assistant, Timeline, and Knowledge Base modules.")

# -------------------------------
# 2️⃣ PROJECT & SUPPORT ASSISTANT
# -------------------------------
with tabs[1]:
    st.header("🤖 Project & Support AI Assistant")
    project_options = [p["name"] for p in projects]
    selected_project_name = st.selectbox("Select Project", project_options)
    project = next(p for p in projects if p["name"] == selected_project_name)
    
    st.subheader("Project Summary")
    summary = f"Project **{project['name']}** for client **{project['client']}** has {len(project['tasks'])} tasks with deadline {project['deadline']}."
    st.info(summary)
    
    st.subheader("Generate Client Reply")
    user_query = st.text_area("Customer / Client Message", "How do I change the language?")
    if st.button("Generate Reply"):
        prompt = f"""
You are a helpful project assistant. The project data is:

{json.dumps(project, indent=2)}

Customer question: {user_query}

Generate a polite, professional reply.
"""
        try:
            with st.spinner("Generating reply..."):
                response = generator(prompt, max_new_tokens=50)
                st.success(response[0]['generated_text'])
        except Exception as e:
            st.error(f"Error generating reply: {str(e)}")

# -------------------------------
# 3️⃣ SMART TIMELINE GENERATOR
# -------------------------------
with tabs[2]:
    st.header("📅 Smart Timeline Generator")
    features = st.text_area("List Project Features (comma separated)", "Landing Page, API, Dashboard")
    complexity = st.slider("Complexity (1-5)", 1, 5, 3)
    urgency = st.slider("Urgency (1-5)", 1, 3, 2)
    
    if st.button("Generate Timeline"):
        base_days = complexity * 3
        urgency_factor = 1.5 if urgency >= 4 else 1.0
        total_days = int(base_days * urgency_factor)
        start_date = datetime.today()
        end_date = start_date + timedelta(days=total_days)
        
        st.write(f"**Estimated Delivery Date:** {end_date.strftime('%Y-%m-%d')}")
        
        # Phase-wise breakdown
        phases = ["Planning", "Design", "Development", "Testing", "Deployment"]
        phase_days = total_days // len(phases)
        timeline = []
        for i, phase in enumerate(phases):
            phase_start = start_date + timedelta(days=i*phase_days)
            phase_end = phase_start + timedelta(days=phase_days)
            timeline.append({"Phase": phase, "Start": phase_start, "End": phase_end})
        df_timeline = pd.DataFrame(timeline)
        st.table(df_timeline)
        
        # Gantt chart
        fig = px.timeline(df_timeline, x_start="Start", x_end="End", y="Phase")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 4️⃣ DOCUMENTATION KNOWLEDGE BASE
# -------------------------------
with tabs[3]:
    st.header("📚 Documentation Knowledge Base")
    st.subheader("Ask a question based on documentation")
    query = st.text_input("Customer Question", "How do I change language on HiStudy?")
    
    # Cache docs
    @st.cache_data
    def load_docs():
        doc_texts = []
        doc_links = []
        doc_files = [f for f in os.listdir("docs") if f.endswith(".txt")]
        for file in doc_files:
            with open(f"docs/{file}", "r") as f:
                content = f.read()
                if "Link:" in content:
                    link = content.split("Link:")[1].strip()
                    text = content.split("Link:")[0].strip()
                else:
                    text = content
                    link = ""
                doc_texts.append(text)
                doc_links.append(link)
        return doc_texts, doc_links
    doc_texts, doc_links = load_docs()
    
    # Cache embedding model
    @st.cache_resource
    def load_embed_model():
        return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embed_model = load_embed_model()
    
    # Compute embeddings
    doc_embeddings = embed_model.encode(doc_texts, batch_size=8, convert_to_numpy=True)
    
    # FAISS index
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    if st.button("Search & Generate Reply"):
        try:
            with st.spinner("Searching and generating reply..."):
                query_embedding = embed_model.encode([query], convert_to_numpy=True)
                D, I = index.search(query_embedding, k=1)
                matched_text = doc_texts[I[0][0]]
                matched_link = doc_links[I[0][0]]
                
                prompt = f"""
You are a helpful support assistant. 
Documentation section: {matched_text}

Customer question: {query}

Generate a concise and polite reply including the link if available.
"""
                response = generator(prompt, max_new_tokens=50)
                st.success(response[0]['generated_text'])
                if matched_link:
                    st.markdown(f"[📄 View Documentation]({matched_link})")
        except Exception as e:
            st.error(f"Error in Knowledge Base: {str(e)}")
