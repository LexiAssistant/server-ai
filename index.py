from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from collections import Counter
import nltk
import requests

nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# NLP 모델 로드
nlp = spacy.load('en_core_web_sm')

def extract_keywords(text):
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks]
    return keywords

def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # 3 문장으로 요약
    return ' '.join([str(sentence) for sentence in summary])

def extract_top_keywords(text, n=10):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    word_freq = Counter(words)
    common_words = word_freq.most_common(n)
    return [word for word, freq in common_words]

@app.route('/process_text', methods=['POST'])
def process_text():
    # 외부 URL에서 데이터 가져오기
    try:
        response = requests.get('https://epson.n-e.kr/data')
        response.raise_for_status()  # 요청이 성공하지 않으면 예외 발생
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch data from external source: {str(e)}"}), 500

    data = response.json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    keywords = extract_keywords(text)
    summary = summarize_text(text)
    top_keywords = extract_top_keywords(text)

    return jsonify({"keywords": keywords, "top_keywords": top_keywords, "summary": summary})
