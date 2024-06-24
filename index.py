from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from collections import Counter
import nltk

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

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    # 요청 본문에서 텍스트 데이터 가져오기
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    keywords = extract_keywords(text)
    summary = summarize_text(text)
    top_keywords = extract_top_keywords(text)

    return jsonify({"keywords": keywords, "top_keywords": top_keywords, "summary": summary})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
