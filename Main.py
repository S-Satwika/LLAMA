import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import requests
import re
import pandas as pd
import evaluate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Define LLaMA 2 API
def get_llama2_response(prompt):
    hf_token = os.getenv("HF_TOKEN")
    api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Gemini API
def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Read PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    return "".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Prompt template
input_prompt_template = """
Hey Act Like a skilled or very experienced ATS(Application Tracking System)
with a deep understanding of tech field, software engineering, data science, data analyst,
and big data engineer. Your task is to evaluate the resume based on the given job description.
The job market is very competitive and you should provide best assistance for improving the resumes.
Assign the percentage Matching based on JD and missing keywords with high accuracy.

Please follow this format strictly:
JD Percentage Match: <percent>
Matching Keywords:
- <keyword1>
- <keyword2>
Missing Keywords:
- <keyword1>: short reason
- <keyword2>: short reason
Profile Summary:
<summary or note if missing>
Final Recommendations:
- <point1>
- <point2>

resume: {text}
description: {jd}
"""

# Extract match %
def extract_match_percentage(text):
    match = re.search(r"(?:JD\s*)?Percentage\s*Match\s*[:\-]?\s*(\d{1,3})", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0

# Streamlit UI
st.sidebar.title("AI-Powered Screening Engine for Modern Recruitment")
model_choice = st.sidebar.radio("Choose Model", ("Gemini", "LLaMA 2"))
jd = st.sidebar.text_area("Paste the Job Description", placeholder="Input the Job description for matching")
uploaded_files = st.sidebar.file_uploader("Upload One or More Resumes", type="pdf", accept_multiple_files=True)
submit = st.sidebar.button("Submit")

# Initialize session storage
if "results" not in st.session_state:
    st.session_state.results = []

if submit and uploaded_files and jd:
    results = []
    st.subheader("🤖 AI Review Finished: Smart Suggestions & Match Score Below")

    for file in uploaded_files:
        resume_text = input_pdf_text(file)
        prompt = input_prompt_template.replace("{text}", resume_text).replace("{jd}", jd)

        reference = get_gemini_response(prompt)
        percent_match = extract_match_percentage(reference)

        if model_choice == "Gemini":
            final_response = reference
            results.append({
                "Candidate": file.name,
                "Match %": percent_match,
                "Model": model_choice,
                "Response": final_response
            })
        else:
            hypothesis = get_llama2_response(prompt)

            bleu_score = bleu.compute(predictions=[hypothesis], references=[reference])["bleu"]
            rouge_score = rouge.compute(predictions=[hypothesis], references=[reference])
            bert_score = bertscore.compute(predictions=[hypothesis], references=[reference], lang="en")

            results.append({
                "Candidate": file.name,
                "Match %": percent_match,
                "Model": model_choice,
                "Response": hypothesis,
                "BLEU": round(bleu_score, 4),
                "ROUGE-1": round(rouge_score["rouge1"], 4),
                "ROUGE-L": round(rouge_score["rougeL"], 4),
                "BERTScore (F1)": round(sum(bert_score["f1"]) / len(bert_score["f1"]), 4)
            })

    st.session_state.results = results

# Display results if available
if st.session_state.results:
    tab1, tab2 = st.tabs(["📊 Ranked Candidates", "📐 Evaluation Metrics"])

    with tab1:
        df = pd.DataFrame(st.session_state.results).sort_values(by="Match %", ascending=False)
        st.dataframe(df[["Candidate", "Match %", "Model"]])

        with st.expander("📋 View Detailed Evaluation for Each Candidate"):
            for row in df.itertuples():
                st.markdown(f"#### {row.Candidate} - Match: {row._2}%")
                st.write(row.Response)
                st.markdown("---")

    with tab2:
        if model_choice == "LLaMA 2":
            metrics_df = pd.DataFrame(st.session_state.results)[
                ["Candidate", "BLEU", "ROUGE-1", "ROUGE-L", "BERTScore (F1)"]
            ]
            st.markdown("### 📐 Model Evaluation Metrics (LLaMA 2 vs Gemini Reference)")
            st.dataframe(metrics_df)

            st.markdown("#### 📊 BLEU Score")
            st.bar_chart(metrics_df.set_index("Candidate")[["BLEU"]])

            st.markdown("#### 📊 ROUGE-1 Score")
            st.bar_chart(metrics_df.set_index("Candidate")[["ROUGE-1"]])

            st.markdown("#### 📊 ROUGE-L Score")
            st.bar_chart(metrics_df.set_index("Candidate")[["ROUGE-L"]])

            st.markdown("#### 📊 BERTScore (F1)")
            st.bar_chart(metrics_df.set_index("Candidate")[["BERTScore (F1)"]])
        else:
            st.info("Evaluation metrics are only available for LLaMA 2 (compared against Gemini).")
