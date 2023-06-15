from flask import Flask, request, render_template, jsonify
import re
from bs4 import BeautifulSoup
import requests
import urllib.parse
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from collections import Counter

app = Flask(__name__)

def read_article(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(file_name, top_n=2):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = read_article(file_name)
    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    for i in range(min(top_n, len(ranked_sentence))):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    summary = ". ".join(summarize_text)
    return summary

def get_summary_from_response(response):
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')

    unwanted_patterns = [
            r'\d{1,2}-[A-Za-z]{3}-\d{4}',  # Date pattern like 22-Mar-2023
            r'\d{1,2}-[A-Za-z]{3}-\d{2}',  # Date pattern like 22-Mar-23
            r'\d{1,2}:[0-5]\d\s?(?:AM|PM)',  # Time pattern like 10:30 AM or 3:45PM
            r'\d{1,2}:[0-5]\d',  # Time pattern like 10:30 or 3:45
            r'[\d-]+[Tt]\d+:[0-5]\d',  # Date and time pattern like 2022-01-01T08:30
            r'[A-Za-z]+\s\d{1,2},\s\d{4}',  # Date pattern like March 22, 2023
            r'\d{1,2}\s[A-Za-z]+\s\d{4}',  # Date pattern like 22 March 2023
            r'[A-Za-z]+\s\d{1,2}\s\d{4}',  # Date pattern like March 22 2023
            r'\d{1,2}\s[A-Za-z]{3,}\s\d{2,4}',  # Date pattern like 22nd March 23
            r'[A-Za-z]+\s\d{1,2}\,\s\d{2,4}',  # Date pattern like March 22,23
            r'\d{1,2}\s[A-Za-z]{3,}\.\s\d{4}',  # Date pattern like 22nd March. 2023
        ]

    paragraph = ''
    for result in search_results:
        paragraph += result.text

    for pattern in unwanted_patterns:
        paragraph = re.sub(pattern, '', paragraph)

    with open('paragraph.txt', 'w', encoding="utf-8") as file:
        file.write(paragraph)

    summary = generate_summary('paragraph.txt', 3)

    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        query = request.form.get('query')
        query = urllib.parse.quote_plus(query)
        print("Query is: " + query)
        url = 'https://google.com/search?q=' + query 
        response = requests.get(url)
        summary = get_summary_from_response(response)
        replace_specialchar1 = summary.replace("...",".");
        replace_specialchar2 = replace_specialchar1.replace("·","")
        replace_specialchar3 = replace_specialchar2.replace(";","\n")
        duplicate = replace_specialchar3.replace(":", " : ")
        clean_text = ' '.join(dict.fromkeys(duplicate.split()))
        print(clean_text)
        return jsonify(summary=clean_text)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
