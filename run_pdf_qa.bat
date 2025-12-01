@echo off
cd /d D:\AI_PROJECT_ML_LLM\PDF_AGENT_ASSISTANT
python pdf_qa_assistant.py --all-pdfs --persist .\index --llm ollama --ollama-model mistral --embedder hf --no-input --show-sources
pause
