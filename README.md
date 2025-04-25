#  ZeuS AI – RAG Powered Multilingual QA System

ZeuS AI is an advanced **Retrieval-Augmented Generation (RAG)** system built for answering academic or general questions from PDFs using Google Gemini 1.5 Flash and HuggingFace models. It supports **multi-language translation**, **semantic reasoning**, **structured context retrieval**, and **web augmentation** via Tavily.

---

##  Features

-  **Gemini 1.5 Flash** integration for high-performance reasoning
-  **PDF Parsing** with automatic section extraction and semantic alignment
-  **Multilingual Support** with real-time translation to/from 100+ languages
-  **Summarization** with BART/DistilBART models
-  **Semantic Search** using SentenceTransformers + FAISS Vector Store
-  **Context Detection** and automatic section-page reference
-  **Web Search Integration** using Tavily API for enhanced answers
-  **Toxic/Unsafe Input Detection** via RoBERTa classifier
-  **Colorful CLI Interface** for easy tracking and result clarity
-  **Structured Output** includes: section name, page numbers, original answer, summary, translation, and URLs
- **RAG Structure Supprotiveness**: Supports for any given PDF file not only the given Grade 11 History Text Book
---

## How to Run ZeuS AI

Follow these steps to set up and run the program locally.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/zeus-ai.git
cd zeus-ai
```

### 2. Install 'reuqirements.txt'

```bash
For Linux: pip3 install -r requirements.txt
For Windows: pip install -r requirements.txt
For MacOS: pip3 install -r requirements.txt
```
### 3. Running the program

```bash
For Linux: python3 fmc_1.py
For Windows: python fmc_1.py
For MacOS: python3 fmc_1.py
```

### System Requirements (Minimum)
---
- **RAM**: Atleast 8GB or more
- **Disk Space**: Atleast 10GB or more
- **CPU**: Atleast i5 8th gen or better processor
- **GPU**: Atleast 4GB or better GPU
---


Developed by TEAM@ZeuS — W.H.Tharusha Rasanjana & P.Y.Isuru Kalhara De Silva.
