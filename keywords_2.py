from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# 데이터베이스 연결 설정
db_config = {
    'user': 'admin',
    'password': 'ewhathon',
    'host': 'ewhathon-db.cp66uoq4s165.ap-northeast-2.rds.amazonaws.com',
    'database': 'ewhathon',
}

# 텍스트 전처리 함수
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 키워드 추출 함수
def extract_keywords(description, keywords, num_keywords=3):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    tfidf_matrix = tfidf_vectorizer.fit_transform([description] + keywords)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    sorted_indices = np.argsort(similarities)[::-1]
    top_keywords_indices = sorted_indices[:num_keywords]
    return [keywords[i] for i in top_keywords_indices]

@app.route('/extractKeywords', methods=['POST'])
def handle_extract_keywords():
    data = request.json
    description = data['description']
    # 추출할 키워드 리스트
    keywords_list = ['신나는', '잔잔한', '힐링', '자극적', '비판적', '감각적', '우아한', '유익한', '창의적',
                    '혼자도_환영', '볼거리_많은', '친구와_함께', '연합_이벤트', '친구_만들기',
                    '홍보_행사', '현장예매_가능', '시즌_이벤트', '외부인_환영']
    processed_text = preprocess_text(description)
    suggested_keywords = extract_keywords(processed_text, keywords_list)
    return jsonify(suggested_keywords)

if __name__ == '__main__':
    app.run(debug=True)
