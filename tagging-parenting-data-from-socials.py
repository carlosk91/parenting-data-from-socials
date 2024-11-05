import openai
import numpy as np
import os
from dotenv import load_dotenv
import psycopg2
import json

# Constants
TAG_THRESHOLD = 0.77
MODEL_VERSION = "v1.0.0"
TAG_VERSION = "v1.0.0"

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class DatabaseOperations:
    def __init__(self):
        self.db = self.connect_to_postgres()

    def connect_to_postgres(self):
        dbhost = os.getenv("PG_DATABASE_HOST")
        dbname = os.getenv("PG_DATABASE_NAME")
        user = os.getenv("PG_USER")
        password = os.getenv("PG_PASSWORD")
        return psycopg2.connect(f"dbname='{dbname}' user='{user}' host='{dbhost}' password='{password}'")

    def get_keywords(self):
        keywords = {}
        with self.db.cursor() as cursor:
            cursor.execute('''
            SELECT t.social_content_tag_id, t.tag_name, k.keyword 
            FROM tag_keywords k JOIN social_content_tags t ON 
                k.social_content_tag_id = t.social_content_tag_id
            WHERE tag_version = %s
            ''', (TAG_VERSION,))
            for tag_id, tag_name, keyword in cursor.fetchall():
                if tag_name not in keywords:
                    keywords[tag_name] = {"tag_id": tag_id, "keywords": []}
                keywords[tag_name]["keywords"].append(keyword)
        return keywords

    def get_texts(self):
        with self.db.cursor() as cursor:
            cursor.execute(
                '''
                SELECT reddit_post_id as content_id, content as text, 'post' as content_type
                FROM reddit_posts
                UNION
                SELECT reddit_comment_id as content_id, body as text, 'comment' as content_type
                FROM reddit_comments
                ''')
            return cursor.fetchall()

    def insert_features(self, content_id, model_version, features):
        with self.db.cursor() as cursor:
            try:
                cursor.execute(
                    '''
                    INSERT INTO social_content_features (content_id, tag_model_version, features)
                    VALUES (%s, %s, %s)
                    ''', (content_id, model_version, json.dumps(features))
                )
                self.db.commit()
            except Exception as e:
                print(f"Error inserting features: {e}")
                self.db.rollback()

    def insert_tags(self, content_id, content_type, tag_name, tag_model_version, keywords):
        tag_info = keywords.get(tag_name)
        if not tag_info:
            print(f"Tag info not found for tag name: {tag_name}")
            return

        tag_id = tag_info["tag_id"]
        with self.db.cursor() as cursor:
            try:
                cursor.execute('''
                    INSERT INTO reddit_content_tags (reddit_content_id, reddit_content_type, social_content_tag_id, 
                        tag_model_version)
                    VALUES (%s, %s, %s, %s)
                    ''', (content_id, content_type, tag_id, tag_model_version))
                self.db.commit()
            except Exception as e:
                print(f"Error inserting tag: {e}")
                self.db.rollback()


# Utility functions
def get_ada_embeddings(texts):
    processed_texts = [text.replace("\n", " ") if isinstance(text, str) else str(text) for text in texts]
    response = openai.embeddings.create(input=processed_texts, model="text-embedding-ada-002")
    return [embedding.embedding for embedding in response.data]


def create_keyword_vectors(keywords):
    return {category: get_ada_embeddings(info["keywords"]) for category, info in keywords.items()}


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def calculate_scores_per_text(text_vec, keyword_vectors):
    feature_scores = {}
    tags_to_insert = []

    for tag, vectors in keyword_vectors.items():
        tag_scores = [cosine_similarity(text_vec, np.array(vec)) for vec in vectors if
                      text_vec.shape == np.array(vec).shape]
        max_score = max(tag_scores, default=0)
        feature_scores[tag] = max_score
        if max_score >= TAG_THRESHOLD:
            tags_to_insert.append(tag)

    return feature_scores, tags_to_insert


# Main function
def main():
    db_ops = DatabaseOperations()
    keywords = db_ops.get_keywords()
    keyword_vectors = create_keyword_vectors(keywords)
    texts = db_ops.get_texts()

    for content_id, text, content_type in texts:
        text_vec = np.array(get_ada_embeddings([text])[0])
        feature_scores, tags_to_insert = calculate_scores_per_text(text_vec, keyword_vectors)

        for tag in tags_to_insert:
            db_ops.insert_tags(content_id, content_type, tag, MODEL_VERSION, keywords)

        db_ops.insert_features(content_id, MODEL_VERSION, feature_scores)


if __name__ == "__main__":
    main()
