# LLAMA
"LLaMA-Powered LLMs for Smarter, Automated Resume  Screening", presented an opportunity to explore how large language models (LLMs), particularly  LLaMA-2-7B, can be adapted to modernize and automate the recruitment process. 

The primary goal was to create a robust system that could intelligently screen resumes, compare them with job descriptions, and provide meaningful feedback moving beyond traditional keyword-based applicant tracking systems.

Fine-tuning a large model like LLaMA-2-7B using QLoRA required an in-depth understanding of quantization techniques, GPU memory optimization, and low-rank adaptation strategies. Setting up the fine-tuning environment, managing limited computational resources, and preparing a domain-specific dataset posed significant hurdles. Additionally, deploying the model to Hugging Face and integrating it within a Streamlit-based interface.

Comparative analysis using Gemini Flash brought further challenges in evaluation—requiring the use of NLP evaluation metrics like BLEU, ROUGE-1, ROUGE-L, and BERTScore. 
