####################################################################################################################################################################
# Created By TEAM@ZeuS
# Programmed By W.H.Tharusha Rasanjana & P.Y.Isuru Kalhara De Silva
# RAG Architecture By: W.H.Tharusha Rasanjana
# Programmed By: W.H.Tharusha Rasanjana
# Further Enhancements: P.Y.Isuru Kalahara De Silva
# Enhancements Included By: W.H.Tharusha Rasanjana
# Main Model ID: Google-Gemini-1.5-Flash
# Other Models[All from huggingface.co]:
# [1] Summarization Tasking
#       bart-large-cnn (Facebook) - 406M Parameters - Available in core-models directory
#       distilbart-cnn-12-6 (sshleifer) - 139M Parameters - Active in here
# [2] Language Translation
#       Many Languages to English (Supported over 200 Languages)
#         opus-mt-mul-en (Helenski-NLP) - 298M Parameters - Active in here
#       English to Many Languages (100 Languages including Sinhala, Tamil, Hindi and else)
#         m2m100_418M (Facebook) - 418M Parameters - Active in here
# [3] Semantic Similarity Checking
#       all-mpnet-base-v2 (SentenceTransformers) - 109M Parameters - 768 Dimensional Embeddings - Active in here
#       all-MiniLM-L6-v2  (SentenceTransformers) - 22M Parameters - 384 Dimensional Embedding - Available in core-models directory
# [4] Violated Questions Checking
#       roberta_toxicity_classifier (s-nlp) - 125M - Active in here
#################################################################################################################################################################
from langchain_community.document_loaders import PyPDFLoader
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_community.vectorstores import FAISS # Facebook AI Similarity Search
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain # Evaluation module
from tavily import TavilyClient # AI search agent
import trafilatura # Advanced Custom URL based data extraction
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity # Cosine similarity check
from spacy_langdetect import LanguageDetector # Check the language for given input
from spacy.language import Language
from safetensors import safe_open # .safetensors model opener
import torch # For probability scanning and others
import json # To gain data from json files(question IDs and questions), To save data into json files (like: submission)
import pprint # Data objects display
import spacy
import re
import numpy as np
import time
import os
import sys
import fitz  # PyMuPDF for PDF operations
import re # Regex check
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
from threading import Thread # For multiprocessing


class clrs(): # Color Initialization Class
    cl0 = Fore.WHITE
    cl1 = Fore.GREEN
    cl2 = Fore.YELLOW
    cl3 = Fore.BLUE
    cl4 = Fore.MAGENTA
    cl5 = Fore.CYAN
    cl6 = Fore.RED
    cb = Style.BRIGHT
    cd = Style.DIM

class symbs(): # Special Symbol Markings Initialization Class
    s1 = "[+]"
    s2 = "[-]"
    s3 = "[!]"
    s4 = "[=]"
    sw = "[WARNING]"
    s5 = f"{clrs.cl0}[{clrs.cl1}+{clrs.cl0}]{clrs.cl0}" # Symbols with colors
    s6 = f"{clrs.cl2}[{clrs.cl1}+{clrs.cl2}]{clrs.cl0}"
    s7 = f"{clrs.cl0}[{clrs.cl3}+{clrs.cl0}]{clrs.cl0}"
    s8 = f"{clrs.cl0}[{clrs.cl4}+{clrs.cl0}]{clrs.cl0}"
    s9 = f"{clrs.cl0}[{clrs.cl1}-{clrs.cl0}]{clrs.cl0}"
    s10 = f"{clrs.cl2}[{clrs.cl1}-{clrs.cl2}]{clrs.cl0}"
    s11 = f"{clrs.cl0}[{clrs.cl3}-{clrs.cl0}]{clrs.cl0}"
    s12 = f"{clrs.cl0}[{clrs.cl4}-{clrs.cl0}]{clrs.cl0}"
    s9 = f"{clrs.cl0}[{clrs.cl1}={clrs.cl0}]{clrs.cl0}"
    s10 = f"{clrs.cl2}[{clrs.cl1}={clrs.cl2}]{clrs.cl0}"
    s11 = f"{clrs.cl0}[{clrs.cl3}={clrs.cl0}]{clrs.cl0}"
    s12 = f"{clrs.cl0}[{clrs.cl4}={clrs.cl0}]{clrs.cl0}"
    ssw = f"{clrs.cl0}[{clrs.cl6}WARNING{clrs.cl0}]{clrs.cl0}"
    s13 = f"{clrs.cl0}[{clrs.cl6}!{clrs.cl0}]{clrs.cl0}"
    s14 = f"{clrs.cl2}[{clrs.cl6}!{clrs.cl2}]{clrs.cl0}"
    so = f"{clrs.cl0}[{clrs.cl1}OVERRIDE LV1{clrs.cl0}]{clrs.cl0}"
    so2 = f"{clrs.cl0}[{clrs.cl1}OVERRIDE LV2{clrs.cl0}]{clrs.cl0}"
    ss = f"{clrs.cl0}[{clrs.cl1}SUCCESS{clrs.cl0}]{clrs.cl0}"
    si = f"{clrs.cl2}[{clrs.cl3}INFO{clrs.cl2}]{clrs.cl0}"

class artwork():
    aw = f"""{Fore.GREEN+Style.BRIGHT}
░▒▓████████▓▒░▒▓████████▓▒░░▒▓██████▓▒░░▒▓██████████████▓▒░       ░▒▓████████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░ 
   ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░             ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
   ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░           ░▒▓██▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
   ░▒▓█▓▒░   ░▒▓██████▓▒░ ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░         ░▒▓██▓▒░  ░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
   ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░       ░▒▓██▓▒░    ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░      ░▒▓████████▓▒░▒▓████████▓▒░░▒▓██████▓▒░░▒▓███████▓▒░                                                                                                                      
{Fore.WHITE}(Artwork By Team@ZeuS)
"""

global PDF_FILE, QUERY_FILE, OUTPUT_FILE, SECTION_JSON_FILE,VECTOR_INDEX,API_KEY,MODEL_ID,TRANSLATOR_ID, VIOLATION_CHECK_ID
# Set below values correctly
PDF_FILE = "g11tb.pdf" # Set the PDF path
QUERY_FILE  = "data.json" # Set the Query path
OUTPUT_FILE  = "submission.json" # Set the output path
SECTION_JSON_FILE = "structured_sections.json" # Set the structured_sections.json file's path
VECTOR_INDEX = "faiss_index"
API_KEY = "AIzaSyBpRMAtgCkb1HVqIV4WSlDPQuWwU3v9Wn4" # Set your Google-Gemini-Flash-1.5 API key
MODEL_ID = "gemini-1.5-flash" # Set your desired model ID here
SEMANTIC_ID = "./core-models/simc_m" # Similirity checker's model path
TRANSLATOR_ID = "./core-models/dtranslator_m" # Translator's model path (Supports for 100 direct language translations including Sinhala, Tamil, Hindi and else)
VIOLATION_CHECK_ID = "./core-models/vcheck_m" # Violation checker of the words
MODEL_TEMPERATURE = 0.2 # Set the model temperature to here
DTIME = 5 # set API request delay time
TOP_K = 1 # set as the TOP_K value you needed depeding on the value the title ranking would be change
os.environ["TAVILY_API_KEY"] = "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" # Set with your valid tavily API key
# !!!  NOTE  !!! 
# When the model temperature > 0.5 it's going to be creative tries not to be concerned about the given cotext more
# When the model themperature = 0.5 it's going to be balanced between concerned and being creative
# Whenever the model temperature < 0.5 it's going to be more concerned except being creative, means more factual to the given sources which we needed in this case

@Language.factory("language_detector") # Set the language detection decorator
def langdetect(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector", last=True)

def check_eng(text):
    doc = nlp(text)
    return doc._.language['language'] != 'en' # Check if the given question's language is in english or not

def detect_language(text):
    doc = nlp(text)
    return doc._.language

def font_divs():
    fsizes = []
    fcount = []
    fcounter = Counter()
    for pagel in extract_pages(PDF_FILE):
        for el in pagel:
            if isinstance(el, LTTextContainer):
                for textl in el:
                    if isinstance(textl, LTTextLine):
                        for char in textl:
                            if isinstance(char, LTChar):
                                fcounter[round(char.size, 1)] += 1
    for size, count in fcounter.most_common(10):
        fsizes.append(size)
        fcount.append(count)
    # Getting the highest letter count as the normally written letters and by that we returns the fsize[max of index of letter count] as the font size
    return fsizes[fcount.index(max(fcount))]


def sectorize(PDF_FILE):
    pages = PyPDFLoader(PDF_FILE).load()
    for i, doc in enumerate(pages):
        doc.metadata["page"] = i + 1
        doc.metadata["section"] = "Unknown"
    split_dt = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    dcs = []
    for page in pages:
        chunks = split_dt.split_documents([page])
        for chunk in chunks:
            chunk.metadata["section"] = page.metadata.get("section", "Unknown")
            chunk.metadata["page"] = page.metadata.get("page", "N/A")
            dcs.append(chunk)
    return dcs

def retry_with_direct_context(llm, prompt_template, question, context): # None Return Recalling Funtion(NRRF)
    prompt_text = prompt_template.format(context=context, question=question)
    result = llm.invoke([{"role": "user", "content": prompt_text}])
    return result.content.strip()

def general_cont(llm,prompt_template,question,context):
    prompt_text = prompt_template.format(question=question, context=context)
    result = llm.invoke([{"role": "user", "content": prompt_text}])
    return result.content.strip()

# Reasoning Based Prompts
# By the below function depeding on the style of the question We will give different type of engineered prompt to Gemini-1.5-Flash
def choose_prompt(question: str): # String type
    structured_prompt = PromptTemplate(input_variables=["context", "question"],
    template="""
You are an expert history assistant. Carefully analyze the question and answer every part of it using only the context provided below. 
Even if the reasoning is not explicitly mentioned, use historical inference to provide a full explanation.
If the question has multiple parts (e.g., causes and effects, implementation and failure), ensure each is answered thoroughly.

Format your response clearly using bullet points or structured paragraphs when appropriate.

Question: {question}

Context:
{context}

Answer:
"""
)

    factual_prompt = PromptTemplate(input_variables=["context", "question"],
    template="""
You are an expert history assistant. Using only the context below, answer the question as factually and completely as possible. 
If the question asks for a list (e.g., reforms, reasons), include all key points mentioned.

Question: {question}

Context:
{context}

Answer:
"""
    )

    multi_part_keywords = [
        "why and how", "why did it", "causes and effects", "implementation and failure",
        "reasons and outcomes", "what and why", "consequences and significance", "what were main demands of","what were major demands of","major contributions",
        "compare and contrast","differences and similarities","advantages and disadvantages","factors and results","how and why",
        "in what ways","role and impact","impact and legacy","explain and evaluate","who and why","purpose and outcome","methods and results",
        "key events and effects","intended and actual","goals and failures",
    ]
    lower_q = question.lower()
    for phrase in multi_part_keywords:
        if phrase in lower_q or ("why" in lower_q and "fail" in lower_q):
            return structured_prompt
    return factual_prompt

def extract_page_range(text):
    match = re.search(r"\(From page:\s*(\d{1,3})\s*-\s*(\d{1,3})\)", text)
    if match:
        start = int(match.group(1)) - 10
        end = int(match.group(2)) - 10
    else:
        match = re.search(r"\(From page:\s*(\d{1,3})\)", text) # Try single page format: (From page: 27)
        if match:
            start = int(match.group(1)) - 10
            end = start
        else:
            return "?"
    start = max(start, 1) # Make sure values are at least 1
    end = max(end, 1)
    return f"{start}-{end}" if start != end else f"{start}"

def rta(answer):
    model = SentenceTransformer(SEMANTIC_ID)
    SECTION_PATH = SECTION_JSON_FILE
    TOP_K = 1
    with open(SECTION_PATH, "r", encoding="utf-8") as f:
        sections = json.load(f)
    titles = list(sections.keys())
    contents = list(sections.values())
    section_embeddings = model.encode(titles, convert_to_tensor=True)
    section_embeddings2 = model.encode(contents, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    cosine_scores_title = util.cos_sim(answer_embedding, section_embeddings)[0]
    cosine_scores_content = util.cos_sim(answer_embedding, section_embeddings2)[0]

    # === Get top match from title and content
    best_idx_title = int(cosine_scores_title.argmax())
    best_idx_content = int(cosine_scores_content.argmax())

    best_title = titles[best_idx_title]
    best_score_title = float(cosine_scores_title[best_idx_title])
    best_content = contents[best_idx_title]

    best_title2 = titles[best_idx_content]
    best_score_content = float(cosine_scores_content[best_idx_content])
    best_content2 = contents[best_idx_content]
    if best_score_title > best_score_content:
        return sections[best_title]
    else:
        return sections[best_title2]

def get_section(SECTION_JSON_FILE,answer,SEMANTIC_ID):
    print(f"{symbs.si} Assemblying the structured data...")
    with open(SECTION_JSON_FILE, "r", encoding="utf-8") as f:
        sections = json.load(f)
    titles = list(sections.keys())
    contents = list(sections.values())
    titles = list(sections.keys()) # Sections of the given PDF
    contents = list(sections.values()) # The contents under those sections
    print(f"{symbs.si} Setting up the model => Model ID :::> {SEMANTIC_ID} ...")
    model_s = SentenceTransformer(SEMANTIC_ID)
    print(f"{symbs.si} Encoding the structured data...")
    sec_emb = model_s.encode(contents, convert_to_tensor=True)
    title_emb = model_s.encode(titles, convert_to_tensor=True)
    print(f"{symbs.si} Encoding the answers...")
    ans_emb = model_s.encode(answer, convert_to_tensor=True)
    cosine_ans = util.cos_sim(ans_emb, sec_emb)[0] # Get the cosine scores with contents embeded
    cosine_ans2 = util.cos_sim(ans_emb, title_emb)[0] # Get the cosine scores with titles embeded
    best_idx = int(cosine_ans.argmax())
    best_idx2 = int(cosine_ans2.argmax())
    best_title = titles[best_idx] # Context (Section based) detection
    best_title2 = titles[best_idx2]
    best_content = contents[best_idx] # Page detection
    best_content2 = contents[best_idx2]
    page_range1 = extract_page_range(best_content)
    pr1 = page_range1.split("-") # Splits data by the range seperator
    if len(pr1) > 1:
        p1,p2 = pr1[0],pr1[1]
    else:
        p1,p2 = pr1[0],pr1[0]
    page_range2 = extract_page_range(best_content2)
    pr2 = page_range2.split("-")
    if len(pr2) > 1:
        p3,p4 = pr2[0],pr2[1]
    else:
        p3,p4 = pr2[0],pr2[0]
    r1,r2 = [p1,p2],[p3,p4]
    # Return's the real referred context with a chosen of three states depeding on the cosine similarity values
    if best_idx > best_idx2:
        return best_title,r1
    if best_idx == best_idx2:
        return best_title,r1
    if best_idx < best_idx2:
        return best_title2,r2

def summarizer(data):
    SUMPATH = "./core-models/summarizer_m"
    sum_tokenizer  = AutoTokenizer.from_pretrained(SUMPATH)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMPATH)
    summarizer_m = pipeline("summarization", model=sum_model, tokenizer=sum_tokenizer)
    dtokens = sum_tokenizer.encode(data, truncation=True)
    dlen = len(dtokens)
    maxlim = int(3/4*(dlen)) # Length dependant(Conveted into token sizes) summarizes
    minlen = int(1/4*(dlen))
    sumup = summarizer_m(data, max_length=maxlim, min_length=minlen, do_sample=False)
    return sumup[0]["summary_text"]

# Via the WWW search if the output is not valid and also for further information
def net_search(adck):
    allowed_urls = [
        "https://kids.nationalgeographic.com/history/article/wright-brothers",
        "https://en.wikipedia.org/wiki/Wright_Flyer",
        "https://airandspace.si.edu/collection-objects/1903-wright-flyer/nasm_A19610048000",
        "https://en.wikipedia.org/wiki/Wright_brothers",
        "https://spacecenter.org/a-look-back-at-the-wright-brothers-first-flight/",
        "https://udithadevapriya.medium.com/a-history-of-education-in-sri-lanka-bf2d6de2882c",
        "https://en.wikipedia.org/wiki/Education_in_Sri_Lanka",
        "https://thuppahis.com/2018/05/16/the-earliest-missionary-english-schools-challenging-shirley-somanader/",
        "https://www.elivabooks.com/pl/book/book-6322337660",
        "https://quizgecko.com/learn/christian-missionary-organizations-in-sri-lanka-bki3tu",
        "https://en.wikipedia.org/wiki/Mahaweli_Development_programme",
        "https://www.cmg.lk/largest-irrigation-project",
        "https://mahaweli.gov.lk/Corporate%20Plan%202019%20-%202023.pdf",
        "https://www.sciencedirect.com/science/article/pii/S0016718524002082",
        "https://www.sciencedirect.com/science/article/pii/S2405844018381635",
        "https://www.britannica.com/story/did-marie-antoinette-really-say-let-them-eat-cake",
        "https://genikuckhahn.blog/2023/06/10/marie-antoinette-and-the-infamous-phrase-did-she-really-say-let-them-eat-cake/",
        "https://www.instagram.com/mottahedehchina/p/Cx07O8XMR8U/?hl=en",
        "https://www.reddit.com/r/HistoryMemes/comments/rqgcjs/let_them_eat_cake_is_the_most_famous_quote/",
        "https://www.history.com/news/did-marie-antoinette-really-say-let-them-eat-cake",
        "https://encyclopedia.ushmm.org/content/en/article/adolf-hitler-early-years-1889-1921",
        "https://en.wikipedia.org/wiki/Adolf_Hitler",
        "https://encyclopedia.ushmm.org/content/en/article/adolf-hitler-early-years-1889-1913",
        "https://www.history.com/articles/adolf-hitler",
        "https://www.bbc.co.uk/teach/articles/zbrx8xs"
    ]
    MAXLIM_CONT = 400
    bt = []
    bu = []
    bestdt = []
    tavc = TavilyClient()
    results = tavc.search(query=adck, search_depth="advanced", max_results=100)
    for i, result in enumerate(results["results"]):
        if result["url"] in allowed_urls:
            bt.append(result["title"])
            bu.append(result["url"])
            if len(result["content"]) > MAXLIM_CONT:
                sumup_res = summarizer(result["content"])
                bestdt.append(sumup_res)
            else:
                bestdt.append(result["content"])
    return bt,bu,bestdt # Best title, URL, Summarized data

            

# If the given input is not english, we will use Helsinki-NLP/opus-mt-mul-en model to convert that sepcific language into English
def to_english(text):
    MODEL_PATH = "./core-models/translator_m"
    translator_tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
    translator_model = MarianMTModel.from_pretrained(MODEL_PATH)
    t_inputs = translator_tokenizer([text], return_tensors="pt", padding=True)
    translate = translator_model.generate(**t_inputs)
    return translator_tokenizer.decode(translate[0], skip_special_tokens=True)

def translate(lang,ans,san,gtl,gct):
    lang_tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATOR_ID)
    lang_model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATOR_ID)
    lang_tokenizer.src_lang = "en" # Set the source language as English
    ed1 = lang_tokenizer(ans, return_tensors="pt", padding=True, truncation=True) # Encode the given final 4 parameters
    ed2 = lang_tokenizer(san, return_tensors="pt", padding=True, truncation=True)
    ed3 = lang_tokenizer(gtl, return_tensors="pt", padding=True, truncation=True)
    ed4 = lang_tokenizer(gct, return_tensors="pt", padding=True, truncation=True)
    gt1 = lang_model.generate(**ed1, forced_bos_token_id=lang_tokenizer.get_lang_id(lang)) # Generate the translation data
    gt2 = lang_model.generate(**ed2, forced_bos_token_id=lang_tokenizer.get_lang_id(lang))
    gt3 = lang_model.generate(**ed3, forced_bos_token_id=lang_tokenizer.get_lang_id(lang))
    gt4 = lang_model.generate(**ed4, forced_bos_token_id=lang_tokenizer.get_lang_id(lang))
    t1 = lang_tokenizer.batch_decode(gt1, skip_special_tokens=True)[0] # Decode generated translation data
    t2 = lang_tokenizer.batch_decode(gt2, skip_special_tokens=True)[0]
    t3 = lang_tokenizer.batch_decode(gt3, skip_special_tokens=True)[0]
    t4 = lang_tokenizer.batch_decode(gt4, skip_special_tokens=True)[0]
    return t1,t2,t3,t4 # Returns all the translated data which translated into the user's input (Contains: real answer, summarized answer, net search title, net search content)

def translate_f(lang,gtl,gct):
    lang_tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATOR_ID)
    lang_model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATOR_ID)
    lang_tokenizer.src_lang = "en" # Set the source language as English
    ed1 = lang_tokenizer(gtl, return_tensors="pt", padding=True, truncation=True)
    ed2 = lang_tokenizer(gct, return_tensors="pt", padding=True, truncation=True)
    gt1 = lang_model.generate(**ed1, forced_bos_token_id=lang_tokenizer.get_lang_id(lang))
    gt2 = lang_model.generate(**ed2, forced_bos_token_id=lang_tokenizer.get_lang_id(lang))
    t1 = lang_tokenizer.batch_decode(gt1, skip_special_tokens=True)[0]
    t2 = lang_tokenizer.batch_decode(gt2, skip_special_tokens=True)[0]
    return t1,t2

def check_violation(question,qid):
    vtokenizer = AutoTokenizer.from_pretrained(VIOLATION_CHECK_ID)
    vmodel = AutoModelForSequenceClassification.from_pretrained(VIOLATION_CHECK_ID)
    vinput = vtokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad(): # Getting model predictions (Using probabilities)
        outputs = vmodel(**vinput)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
    violation_classification = ["Toxic", "Severe toxic", "Obscene", "Threat", "Insult", "Identity hate"] # Probability based classification
    for label, prob in zip(violation_classification, probabilities[0]):
        if label == violation_classification[1]:
            if round(prob.item(),4) > 0.75:
                return False,f"{symbs.ssw} Sorry, Your given question '{question}' with got the query id as {qid} contains {label}!"
            else:
                return True,f""

def check_gen(question):
    tfa = []
    cl = [
        "don't based on given context above",
        "without using the given context",
        "search the web to find the latest information.",
        "use current events to answer.",
        "use general knowledge to explain.",
        "go beyond the context if needed.",
        "use AI reasoning where context lacks information.",
        "combine context with your own knowledge.",
        "include facts not in the context if helpful.",
        "you may speculate using common knowledge.",
        "incorporate outside references if context is insufficient.",
        "find updated data if the context is outdated.",
        "look up recent changes or updates online.",
        "explain using broader understanding of the topic.",
        "if missing, infer from general principles.",
        "leverage world knowledge to fill in gaps.",
        "assume reasonable facts beyond the text.",
        "go beyond the document when necessary.",
        "supplement the context with known facts.",
    ]
    for c in cl:
        if c not in question.lower():
            tfa.append(True)
        else:
            tfa.append(False)
    if False in tfa:
        return False
    else:
        return True
    

def updated(question):
    tfr = []
    updt = [
    "latest",
    "new",
    "recent",
    "today",
    "nowadays",
    "currently",
    "as of now",
    "up-to-date",
    "real-time info",
    "trending",
    "news",
    "use external resources if needed.",
    "refer to current events or statistics.",
    "consult your knowledge base if context is limited.",
    ]
    # "last president",
    # "last prime minister",
    # "last minister",
    # "last mayor",
    for c in question.split():
        if c in updt:
            return False
    return True

def process(queries,DTIME,vstore,llm):
    SUMUP_LIM = 360 # Change as you desired
    results = []
    for q in queries:
        ce = check_eng(q["Question"])
        dl = detect_language(q["Question"])
        print(f"{symbs.si} Detected language: {dl['language']}")
        if ce == True:
            question = q["Question"]
        else:
            question = to_english(q["Question"])
        violated, vcontp = check_violation(question,q["Query ID"])
        if violated == False:
            print(vcontp)
        else:
            time.sleep(DTIME)
            qid = q["Query ID"]
            cav = check_gen(question)
            upc = updated(question)
            #print(f"CAV: {cav}")
            if  cav == True and upc == True:
                the_prompt = choose_prompt(question)
                retriever = vstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}) # Retrieve the vector data
                rag_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=True,chain_type_kwargs={"prompt": the_prompt}) # Initialize the RAG chain
                result = rag_chain.invoke({"query": question})
                answer = result["result"].strip()
                if "the question cannot be answered" in answer.lower() or "there is no mention" in answer.lower() or "the provided text does not contain any information" in answer.lower() or " no further details on" in answer.lower():
                    print(f"{symbs.so} Forcing the system...")
                    ta = rta(question) # Context extractor
                    promptr = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""
                        You are an expert history assistant. Using only the context below, answer the question as factually and completely as possible. 
                        If the question asks for a list (e.g., reforms, reasons), include all key points mentioned.

                        Question: {question}

                        Context:
                        {context}

                    Answer:
                    """
                    )
                    rag_chain2 = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=True,chain_type_kwargs={"prompt": promptr})
                    answer2 = retry_with_direct_context(llm, promptr, question, ta)
                    if "the question cannot be answered" not in answer2.lower() and "there is no mention" not in answer2.lower() and "the provided text does not contain any information" not in answer2.lower() and " no further details on" not in answer2.lower():
                        print(f"{symbs.si} Operating a semantic search of given answer...") # Section detection
                        title,cp = get_section(SECTION_JSON_FILE,answer2,SEMANTIC_ID)
                        sans = ''
                        if len(answer2) > SUMUP_LIM:
                            print(f"{symbs.si} Summarizing the given answer...")
                            sans = summarizer(answer2)
                        print(f"{symbs.si} Net-searching from real world datasets...")
                        gt,gu,gc = net_search(question)
                        if len(gu) > 0:
                            if dl['language'] != 'en':
                                orans,sansl,gtx,gcx = translate(dl,answer2,sans,gt,gc)
                                print(f"{symbs.si} Question ID: {qid}")
                                print(f"{symbs.si} Question: {question}")
                                print(f"{symbs.ss} Answer(Not Summarized): {orans}")
                                if sans != '':
                                    print(f"{symbs.ss} Answer(Summarized): {sansl}")
                                print(f"{symbs.ss} Section: {title}")
                                if cp[0] != cp[1]:
                                    print(f"{symbs.ss} Pages: {cp[0]} {cp[1]}")
                                else:
                                    print(f"{symbs.ss} Pages: {cp[0]}") 
                                cnt = 1
                                for ctitle, curl, ccont in zip(gtx,gu,gcx):
                                    print(f"{symbs.ss} Title No.{cnt} of {len(gtx)}: {ctitle}")
                                    print(f"{symbs.ss} URL No.{cnt} of {len(gu)}: {curl}")
                                    print(f"{symbs.ss} Found Answer No.{cnt} of {len(gcx)}: {ccont}")
                                    cnt += 1
                                #print(f"{symbs.ss} QID: {qid} |=> {orans} |=> {sansl}|=> {title} |=> {cp} |=> {gtx} |=> {gu} |=> {gcx}") # If the language is not English
                            else:
                                print(f"{symbs.si} Question ID: {qid}")
                                print(f"{symbs.si} Question: {question}")
                                print(f"{symbs.ss} Answer(Not Summarized): {answer}")
                                if sans != '':
                                    print(f"{symbs.ss} Answer(Summarized): {sans}")
                                print(f"{symbs.ss} Section: {title}")
                                if cp[0] != cp[1]:
                                    print(f"{symbs.ss} Pages: {cp[0]} {cp[1]}")
                                else:
                                    print(f"{symbs.ss} Pages: {cp[0]}") 
                                nt = 1
                                for ft, fu, fc in zip(gt,gu,gc):
                                    print(f"{symbs.ss} Title No.{nt} of {len(gt)}: {gt}")
                                    print(f"{symbs.ss} URL No.{nt} of {len(gu)}: {fu}")
                                    print(f"{symbs.ss} Found Answer No.{nt} of {len(gc)}: {fc}")
                                    nt += 1
                        else:
                            print(f"{symbs.si} Question ID: {qid}")
                            print(f"{symbs.si} Question: {question}")
                            print(f"{symbs.ss} Answer(Not Summarized): {orans}")
                            if sans != '':
                                print(f"{symbs.ss} Answer(Summarized): {sans}")
                            print(f"{symbs.ss} Section: {title}")
                            if cp[0] != cp[1]:
                                print(f"{symbs.ss} Pages: {cp[0]} {cp[1]}")
                            else:
                                print(f"{symbs.ss} Pages: {cp[0]}") 
                            print(f"{symbs.si} Net-search status: None")
                    else:
                        print(f"{symbs.so} Forcing the system...")
                        print(f"{symbs.si} Net-searching from real world datasets...")
                        dgt,dgu,dgc = net_search(question)
                        if dl['language'] != 'en': # If the language is not English
                            orans,sansl,gtx,gcx = translate(dl,answer2,sans,dgt,dgc)
                            print(f"{symbs.si} Question ID: {qid}")
                            print(f"{symbs.si} Question: {question}")
                            n = 1
                            for utitle, urlx, ucont in zip(gtx,dgu,gcx):
                                print(f"{symbs.ss} Title No.{n} of {len(gtx)}: {utitle}")
                                print(f"{symbs.ss} URL No.{n} of {len(dgu)}: {urlx}")
                                print(f"{symbs.ss} Found Answer No.{n} of {len(gcx)}: {ucont}")
                                n += 1
                        else: # If the language is English
                            w = 1
                            print(f"{symbs.si} Question ID: {qid}")
                            print(f"{symbs.si} Question: {question}")
                            for gtitle, urlg, gcont in zip(dgt,dgu,dgc):
                                print(f"{symbs.ss} Title No.{w} of {len(dgt)}: {gtitle}")
                                print(f"{symbs.ss} URL No.{w} of {len(dgu)}: {urlg}")
                                print(f"{symbs.ss} Found Answer No.{w} of {len(dgc)}: {gcont}")
                                w += 1
                else: # No Overrides
                    print(f"{symbs.si} Operating a semantic search of given answer...") # Section detection
                    title,cp = get_section(SECTION_JSON_FILE,answer,SEMANTIC_ID)
                    sans = ''
                    if len(answer) > SUMUP_LIM:
                        print(f"{symbs.si} Summarizing the given answer...")
                        sans = summarizer(answer)
                    print(f"{symbs.si} Net-searching from real world datasets...")
                    gt,gu,gc = net_search(question)
                    if len(gu) > 0:
                        if dl['language'] != 'en': # If the language is not English
                            orans,sansl,gtx,gcx = translate(dl,answer,sans,gt,gc)
                            print(f"{symbs.si} Question ID: {qid}")
                            print(f"{symbs.si} Question: {question}")
                            print(f"{symbs.ss} Answer(Not Summarized): {orans}")
                            if sans != '':
                                print(f"{symbs.ss} Answer(Summarized): {sansl}")
                            print(f"{symbs.ss} Section: {title}")
                            if cp[0] != cp[1]:
                                print(f"{symbs.ss} Pages: {cp[0]} {cp[1]}")
                            else:
                                print(f"{symbs.ss} Pages: {cp[0]}")
                            a = 1
                            for qpt, qu, qc in zip(gtx,gu,gcx):
                                print(f"{symbs.ss} [Works As The Section] Title No.{a} of {len(gtx)}: {qpt}")
                                print(f"{symbs.ss} URL No.{a} of {len(gu)}: {qu}")
                                print(f"{symbs.ss} Found Answer No.{a} of {len(gcx)}: {qc}")
                                a += 1
                        # print(f"{symbs.ss} QID: {qid} |=> {orans} |=> {sansl}|=> {title} |=> {cp} |=> {gtx} |=> {gu} |=> {gcx}")
                        else: # If the language is English
                            print(f"{symbs.si} Question ID: {qid}")
                            print(f"{symbs.si} Question: {question}")
                            print(f"{symbs.ss} Answer(Not Summarized): {answer}")
                            if sans != '':
                                print(f"{symbs.ss} Answer(Summarized): {sans}")
                            print(f"{symbs.ss} Section: {title}")
                            if cp[0] != cp[1]:
                                print(f"{symbs.ss} Pages: {cp[0]} {cp[1]}")
                            else:
                                print(f"{symbs.ss} Pages: {cp[0]}")
                            b = 1
                            for rpt, ru, rc in zip(gt,gu,gc):
                                print(f"{symbs.ss} [Works As The Section] Title No.{b} of {len(gt)}: {rpt}")
                                print(f"{symbs.ss} URL No.{b} of {len(gu)}: {ru}")
                                print(f"{symbs.ss} Found Answer No.{b} of {len(gc)}: {rc}")
                                b += 1
                    else:
                        print(f"{symbs.si} Question ID: {qid}")
                        print(f"{symbs.si} Question: {question}")
                        print(f"{symbs.ss} Answer(Not Summarized): {answer}")
                        if sans != '':
                            print(f"{symbs.ss} Answer(Summarized): {sans}")
                        print(f"{symbs.ss} Section: {title}")
                        if cp[0] != cp[1]:
                            print(f"{symbs.ss} Pages: {cp[0]} {cp[1]}")
                        else:
                            print(f"{symbs.ss} Pages: {cp[0]}")
                        print(f"{symbs.si} Net-search status: None")
            else: # In here the Model will freely tell his opinion or use real world cases, So there are no sections and pages
                if cav == False and upc == True:
                    promptg = PromptTemplate(
                    input_variables=["context","question"],
                    template="""
                    Answer to given question freely.

                    Question: {question}

                    Context:
                    {context}

                    Answer:
                    """
                    )
                    retriever = vstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}) # Retrieve the vector data
                    genaral = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=True,chain_type_kwargs={"prompt": promptg})
                    gans = general_cont(llm, promptg, question,"Answer to given question freely.")
                    sans = ''
                    if len(gans) > SUMUP_LIM:
                        print(f"{symbs.si} Summarizing the given answer...")
                        sans = summarizer(gans)
                    print(f"{symbs.si} Net-searching from real world datasets...")
                    gt,gu,gc = net_search(question)
                    if dl['language'] != 'en':
                        orans,sansl,gtx,gcx = translate(dl,gans,sans,gt,gc)
                        print(f"{symbs.si} Question ID: {qid}")
                        print(f"{symbs.si} Question: {question}")
                        print(f"{symbs.ss} Answer(Not Summarized): {gans}")
                        if sans != '':
                            print(f"{symbs.ss} Answer(Summarized): {sansl}")
                        print(f"{symbs.si} Section: Not Found")
                        print(f"{symbs.si} Pages: None")
                        m = 1
                        for wpt, wu, wc in zip(gtx,gu,gcx):
                            print(f"{symbs.ss} [Works As The Section] Title No.{m} of {len(gtx)}: {wpt}")
                            print(f"{symbs.ss} URL No.{m} of {len(gu)}: {wu}")
                            print(f"{symbs.ss} Found Answer No.{m} of {len(gcx)}: {wc}")
                            m += 1
                    else:
                        print(f"{symbs.si} Question ID: {qid}")
                        print(f"{symbs.si} Question: {question}")
                        print(f"{symbs.ss} Answer: {gans}")
                        print(f"{symbs.ss} Answer(Not Summarized): {gans}")
                        if sans != '':
                            print(f"{symbs.ss} Answer(Summarized): {sansl}")
                        print(f"{symbs.si} Section: Not Found")
                        print(f"{symbs.si} Pages: None")
                        p = 1
                        for pt, pu, pc in zip(gt,gu,gc):
                            print(f"{symbs.ss} [Works As The Section] Title No.{p} of {len(gt)}: {pt}")
                            print(f"{symbs.ss} URL No.{p} of {len(gu)}: {pu}")
                            print(f"{symbs.ss} Found Answer No.{p} of {len(gc)}: {pc}")
                            p += 1
                        #print(f"{symbs.ss} QID: {qid} |=> {gans} |=> {gt} |=> {gu} |=> {gc}") # If the language is English
                elif (cav == True and upc == False) or (cav == False and upc == False):
                    print(f"{symbs.si} Net-searching from real world datasets...")
                    kgt,kgu,kgc = net_search(question)
                    if len(kgu) > 0:
                        if dl == 'en':
                            k = 1
                            print(f"{symbs.si} Question ID: {qid}")
                            print(f"{symbs.si} Question: {question}")
                            print(f"{symbs.si} Section: Not Found")
                            print(f"{symbs.si} Pages: None")
                            for ktitle, urlk, kcont in zip(kgt,kgu,kgc):
                                print(f"{symbs.ss} [Works As The Section] Title No.{k} of {len(kgt)}: {ktitle}")
                                print(f"{symbs.ss} URL No.{k} of {len(kgu)}: {urlk}")
                                print(f"{symbs.ss} Found Answer No.{k} of {len(kgc)}: {kcont}")
                                k += 1
                        else:
                            rxt,rxc = translate_f(dl,kgt,kgc)
                            z = 1
                            print(f"{symbs.si} Question ID: {qid}")
                            print(f"{symbs.si} Question: {question}")
                            print(f"{symbs.si} Answer: Not available in the given context.So we've gone for net-search!")
                            print(f"{symbs.si} Section: Not Found")
                            print(f"{symbs.si} Pages: None")
                            for ztitle, urlz, zcont in zip(rxt,kgu,rxc):
                                print(f"{symbs.ss} [Works As The Section] Title No.{z} of {len(rxt)}: {ztitle}")
                                print(f"{symbs.ss} URL No.{z} of {len(kgu)}: {urlz}")
                                print(f"{symbs.ss} Found Answer No.{z} of {len(rxc)}: {zcont}")
                                z += 1
                    else:
                        print(f"{symbs.si} Sorry, We couldn't find anything from the net-search even using the given URL criteria!")
                
def makes(nfz,PDF_FILE):
    doc = fitz.open(PDF_FILE)
    MIN_FONT_SIZE = nfz
    MAX_TITLE_WORDS = 14
    sections = {}
    current_title = None
    buffer = ""
    print(f"{symbs.si} Scaning for all available titles and contents...")
    pgr = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue # Avoids from the data objects like Images

            for l in b["lines"]:
                for s in l["spans"]:
                    text_dt = s["text"].strip()
                    font_dt = s["font"]
                    size_dt = s["size"]
                    if not text_dt:
                        continue
                    is_bold = "Bold" in font_dt or font_dt.endswith("bd") or "Semibold" in font_dt or font_dt.endswith("Semibold")
                    is_title = size_dt >= MIN_FONT_SIZE and len(text_dt.split()) <= MAX_TITLE_WORDS
                    if is_bold and is_title: # Flushes previous title's buffer
                        if current_title and buffer.strip():
                            if current_title not in sections:
                                if len(pgr) > 0:
                                    sections[current_title] = f"(From page: {str(pgr[-1]) + '-' + str(page_num)}) {buffer.strip()}"
                                else:
                                    sections[current_title] = f"(From page: {str(page_num)}) {buffer.strip()}" #str(pgr[-1]) + '-' + 
                                pgr.append(page_num)
                            else:
                                sections[current_title] += " " + buffer.strip()
                        print(f"{symbs.s5} Page {page_num}: '{text_dt}' [Font: {font_dt}, Size: {size_dt}]")
                        current_title = text_dt
                        buffer = ""
                    else:
                        buffer += " " + text_dt
    # Flush again as the final flush
    if current_title and buffer.strip():
        if current_title not in sections:
            sections[current_title] = f"(From page: {page_num}) {buffer.strip()}"
    else:
        sections[current_title] += " " + buffer.strip()
    with open(SECTION_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    return f"{symbs.ss} Data restructed successfully and saved to {SECTION_JSON_FILE} in {os.getcwd()}!"

def main():
    print(artwork.aw)
    print(f"{symbs.si} Splitting the text book...")
    docs = sectorize(PDF_FILE)
    print(f"{symbs.si} Data splitted successfully!")
    print(f"{symbs.si} Finding the font sizes of the given text book...")
    nfz = font_divs()
    print(f"{symbs.si} Successfully extracted the font sizes from the text book!")
    print(f"{symbs.si} Restructuring the given file...")
    structured = makes(nfz,PDF_FILE)
    print(structured)
    print(f"{symbs.si} Embedding the documents...")
    os.environ["GOOGLE_API_KEY"] = API_KEY
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(VECTOR_INDEX)
    print(f"{symbs.si} Setting up the model => Model ID :::> {MODEL_ID} ...")
    # Adding safety settings (Not much enough, So we have added a model something that specialised for that purpose also)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    llm = ChatGoogleGenerativeAI(model=MODEL_ID, temperature=MODEL_TEMPERATURE, safety_settings=safety_settings)
    print(f"{symbs.si} Current model temperature => {MODEL_TEMPERATURE}")
    print(f"{symbs.si} Generating answers...")
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)
    get_res = process(queries,DTIME,vectorstore,llm)
    print(f"{symbs.si} Thank you for using our RAG model!")
    sys.exit()

if __name__ == '__main__':
    main()

# Created By TEAM@ZeuS
