import matplotlib.pyplot as plt
import openai
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
import csv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora


load_dotenv()

keywords = {
    "Time": ["busy schedule", "no time", "time management", "constantly busy", "lack of time", "hectic life",
             "time-poor", "juggling responsibilities", "overwhelmed with tasks", "limited time availability"],
    "Strangers": ["online stranger", "unknown people", "stranger danger", "unsafe interactions", "online safety",
                  "unknown contacts", "stranger interaction", "safety from strangers", "unfamiliar online users",
                  "protecting from strangers"],
    "Content": ["inappropriate games", "adult content", "violent games", "explicit material", "safe content",
                "age-inappropriate content", "mature themes", "graphic content", "restrictive content",
                "content filtering"],
    "Conversations": ["chatting issues", "online communication", "bad language", "managing conversations",
                      "communication skills", "managing online chats", "inappropriate discussions", "online talk",
                      "safe chatting", "handling online interactions"],
    "Guidance": ["parental advice", "guidance needed", "how to guide", "parenting tips", "digital parenting",
                 "seeking parenting advice", "navigating digital parenting", "parental guidance strategies",
                 "guidance for online safety", "child online guidance"],
    "Monitoring": ["account monitoring", "watching over", "privacy concerns", "supervision", "keeping an eye",
                   "tracking online activity", "digital supervision", "overseeing internet use",
                   "monitoring digital footprint", "online behavior monitoring"],
    "Isolation": ["feeling alone", "only one worried", "parental isolation", "lacking support", "lone parenting",
                  "solo parenting challenges", "alone in parenting concerns", "lack of parenting support",
                  "feeling isolated as a parent", "no support in child rearing"],
    "Limits": ["setting boundaries", "restricting access", "age-appropriate limitations", "parental controls",
               "online rules", "gaming limits", "screen time restrictions", "safe online practices"]
}

openai.api_key = os.getenv("OPENAI_API_KEY")
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

sia = SentimentIntensityAnalyzer()

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def get_ada_embeddings(texts):
    processed_texts = [text.replace("\n", " ") if isinstance(text, str) else str(text) for text in texts]
    response = openai.embeddings.create(
        input=processed_texts,
        model="text-embedding-ada-002"
    )
    embeddings = [embedding.embedding for embedding in response.data]
    return embeddings


def get_sentiment(text):
    return sia.polarity_scores(text)


def create_keyword_vectors(keywords):
    keyword_vectors = {}
    for category, phrases in keywords.items():
        keyword_vectors[category] = get_ada_embeddings(phrases)
    return keyword_vectors


def connect_to_postgres():
    dbhost = os.getenv("PG_DATABASE_HOST")
    dbname = os.getenv("PG_DATABASE_NAME")
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    return psycopg2.connect(f"dbname='{dbname}' user='{user}' host='{dbhost}' password='{password}'")


def get_sample_texts(db, limit=50):
    texts = []
    with db.cursor() as cursor:
        try:
            cursor.execute(
                '''
                SELECT content 
                FROM reddit_posts 
                WHERE random()<0.5
                ORDER BY random()
                LIMIT %s''', (limit,))
            texts.extend([row[0] for row in cursor.fetchall()])

            cursor.execute(
                '''
                SELECT body 
                FROM reddit_comments 
                WHERE random()<0.2
                ORDER BY random()
                LIMIT %s''', (limit,))
            texts.extend([row[0] for row in cursor.fetchall()])

        except Exception as e:
            print(f"Error fetching sample texts: {e}")
    return texts


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def create_scores_by_tag(keywords, sample_texts, keyword_vectors):
    scores_by_tag = {tag: [] for tag in keywords.keys()}

    for text in sample_texts:
        text_vec = np.array(get_ada_embeddings([text])[0])

        for tag, vectors in keyword_vectors.items():
            tag_scores = []
            for vec in vectors:
                vec_array = np.array(vec)

                if text_vec.shape != vec_array.shape:
                    print(f"Shape mismatch: text_vec {text_vec.shape}, vec_array {vec_array.shape}")
                else:
                    score = cosine_similarity(text_vec, vec_array)
                    tag_scores.append(score)

            max_score = max(tag_scores, default=0)
            scores_by_tag[tag].append(max_score)

    return scores_by_tag


def get_clean_text(text):
    stop_free = " ".join([i for i in text.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def get_clean_list_of_texts(list_of_texts):
    return [get_clean_text(doc).split() for doc in list_of_texts]


def create_dictionary_and_doc_term_matrix(list_of_texts):
    clean_list_of_texts = get_clean_list_of_texts(list_of_texts)
    dictionary = corpora.Dictionary(clean_list_of_texts)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_list_of_texts]
    return dictionary, doc_term_matrix


def save_to_csv(scores_by_tag, texts, file_name):
    headers = ['Content'] + list(scores_by_tag.keys()) + ['sia']

    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for i, text in enumerate(texts):
            row = [text] + [scores_by_tag[tag][i] for tag in scores_by_tag] + [get_sentiment(text)]
            writer.writerow(row)


def graph_scores_by_graph(scores_by_tag):
    for tag, scores in scores_by_tag.items():
        plt.figure()
        plt.hist(scores, bins=20, alpha=0.7, label=tag)
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.title(f"Score Distribution for '{tag}' Tag")
        plt.legend()
        plt.show()

        print(f"Tag: {tag}")
        print(f"Mean Score: {np.mean(scores)}")
        print(f"Median Score: {np.median(scores)}")
        print(f"Standard Deviation: {np.std(scores)}")


def main():
    keyword_vectors = create_keyword_vectors(keywords)
    db = connect_to_postgres()
    sample_texts = get_sample_texts(db, limit=50)
    scores_by_tag = create_scores_by_tag(keywords, sample_texts, keyword_vectors)
    save_to_csv(scores_by_tag, sample_texts, 'exports/tag_scores.csv')
    graph_scores_by_graph(scores_by_tag)
