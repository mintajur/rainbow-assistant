# rainbow_assistant.py

import streamlit as st
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import os
from openai import OpenAI

# --- CONFIGURATION ---
# Set your OpenAI API key as a Streamlit secret or environment variable
# st.secrets["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- MOCK DATA ---
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

# Mock documentation files
docs_folder = "docs"
os.makedirs(docs_folder, exist_ok=True)
docs_content = {
    "histudy.txt": "# Section: Language Settings\nTo change the site language, go to WPML → Languages → Add Language.\nLink: https://docs.yoursite.com/histudy/language-settings\n",
    "aiwave.txt": "# Section: API Setup\nTo setup AI API, follow the instructions in AIwave dashboard.\nLink: https://docs.yoursite.com/aiwave/api-setup\n"
}
for filename, content in docs_content.items():
    with open(os.path.join(docs_folder, filename), "w") as f:
        f.write(content)

# --- STREAMLIT LAYOUT ---
st.set_page_config(page_title="Rainbow Assistant", layout="wide")
st.title("🌈 Rainbow Assistant — AI Project Manager & Knowledge Base")

tabs = st.tabs(["🏠 Dashboard", "🤖 Project Assistant", "📅 Timeline Generator", "📚 Knowledge Base"])

# ----------------------
# 1️⃣ DASHBOARD
# ----------------------
with tabs[0]:
    st.header("Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Projects", len(projects))
    pending_tasks = sum([len([t for t in p["tasks"] if t["status"] != "Done"]) for p in projects])
    col2.metric("Pending Tasks", pending_tasks)
    upcoming_deadline = min([datetime.strptime(p["deadline"], "%Y-%m-%d") for p in projects])
    col3.metric("Next Deadline", upcoming_deadline.strftime("%Y-%m-%d"))
    
    st.write("Use the tabs to explore Assistant, Timeline, and Knowledge Base modules.")

# ----------------------
# 2️⃣ PROJECT & SUPPORT ASSISTANT
# ----------------------
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        reply = response.choices[0].message.content
        st.success(reply)

# ----------------------
# 3️⃣ SMART TIMELINE GENERATOR
# ----------------------
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
        fig = px.timeline(df_timeline, x_start="Start", x_end="End", y="Phase", color="Phase")
        st.plotly_chart(fig)

# ----------------------
# 4️⃣ DOCUMENTATION KNOWLEDGE BASE
# ----------------------
with tabs[3]:
    st.header("📚 Documentation Knowledge Base")
    st.subheader("Ask a question based on documentation")
    query = st.text_input("Customer Question", "How do I change language on HiStudy?")
    
    if st.button("Search & Generate Reply"):
        # Simple keyword search
        matched_text = ""
        matched_link = ""
        for file in os.listdir(docs_folder):
            path = os.path.join(docs_folder, file)
            with open(path, "r") as f:
                content = f.read()
                if all(word.lower() in content.lower() for word in query.split()):
                    matched_text = content
                    if "Link:" in content:
                        matched_link = content.split("Link:")[1].strip()
                    break
        if matched_text:
            prompt = f"""
You are a helpful support assistant. 
Documentation section: {matched_text}

Customer question: {query}

Generate a concise and polite reply including the link if available.
"""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            reply = response.choices[0].message.content
            st.success(reply)
            if matched_link:
                st.markdown(f"[📄 View Documentation]({matched_link})")
        else:
            st.warning("No matching documentation found.")