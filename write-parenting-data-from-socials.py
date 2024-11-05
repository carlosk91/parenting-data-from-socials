import praw
import os
from dotenv import load_dotenv
import psycopg2
from datetime import datetime

load_dotenv()


def connect_to_reddit_app():
    try:
        return praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"),
                           client_secret=os.getenv("REDDIT_SECRET"),
                           user_agent=os.getenv("REDDIT_USER_AGENT"),
                           username=os.getenv("REDDIT_USER"),
                           password=os.getenv("REDDIT_PASSWORD"))
    except Exception as e:
        print(f"Error connecting to Reddit: {e}")
        raise


def connect_to_postgres():
    dbhost = os.getenv("PG_DATABASE_HOST")
    dbname = os.getenv("PG_DATABASE_NAME")
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    try:
        db = psycopg2.connect(f"dbname='{dbname}' user='{user}' host='{dbhost}' password='{password}'")
    except Exception as e:
        print(f"Unable to connect to the database: {e}")
        raise
    return db


def write_subreddit_to_postgres(db, subreddit):
    with db.cursor() as cursor:
        try:
            cursor.execute(
                """
                INSERT INTO reddit_subreddits (subreddit_id, display_name, subscriber_count, public_description, 
                    creation_date, active_user_count, updated_at) 
                VALUES (%s, %s, %s, %s, %s, %s, current_timestamp)
                ON CONFLICT (subreddit_id) 
                DO UPDATE SET 
                    display_name = EXCLUDED.display_name, 
                    subscriber_count = EXCLUDED.subscriber_count, 
                    public_description = EXCLUDED.public_description, 
                    creation_date = EXCLUDED.creation_date,
                    active_user_count = EXCLUDED.active_user_count,
                    updated_at = EXCLUDED.updated_at
                """, (subreddit.id, subreddit.display_name, subreddit.subscribers, subreddit.public_description,
                      datetime.utcfromtimestamp(subreddit.created_utc), subreddit.active_user_count))
            db.commit()
        except Exception as e:
            print(f"Error inserting subreddit into PostgreSQL: {e}")


def insert_post_search_query(db, post_id, search_query):
    with db.cursor() as cursor:
        try:
            cursor.execute(
                """
                INSERT INTO reddit_post_search_queries (reddit_post_id, search_query, updated_at) 
                VALUES (%s, %s, current_timestamp)
                ON CONFLICT (reddit_post_id, search_query)
                DO UPDATE SET
                     reddit_post_id = EXCLUDED.reddit_post_id, 
                     search_query = EXCLUDED.search_query, 
                     updated_at = EXCLUDED.updated_at
                """, (post_id, search_query))
            db.commit()
        except Exception as e:
            print(f"Error inserting post-search query association: {e}")


def write_posts_to_postgres(db, submission):
    try:
        with db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reddit_posts (reddit_post_id, title, content, score, author, timestamp, updated_at) 
                VALUES (%s, %s, %s, %s, %s, %s, current_timestamp)
                ON CONFLICT (reddit_post_id) 
                DO UPDATE SET 
                    title = EXCLUDED.title, 
                    content = EXCLUDED.content, 
                    score = EXCLUDED.score, 
                    author = EXCLUDED.author, 
                    timestamp = EXCLUDED.timestamp,
                    updated_at = EXCLUDED.updated_at
                """, (submission.id, submission.title, submission.selftext, submission.score,
                      submission.author.name if submission.author else "Deleted",
                      datetime.utcfromtimestamp(submission.created_utc)))
            db.commit()
    except Exception as e:
        print(f"Error writing post to PostgreSQL: {e}")


def write_comments_to_postgres(db, comment, submission):
    try:
        with db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reddit_comments (reddit_comment_id, body, score, author, timestamp, parent_id, 
                    reddit_post_id, updated_at) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, current_timestamp)
                ON CONFLICT (reddit_comment_id) 
                DO UPDATE SET 
                    body = EXCLUDED.body, 
                    score = EXCLUDED.score, 
                    author = EXCLUDED.author, 
                    timestamp = EXCLUDED.timestamp, 
                    parent_id = EXCLUDED.parent_id, 
                    reddit_post_id = EXCLUDED.reddit_post_id,
                    updated_at = EXCLUDED.updated_at
                """, (comment.id, comment.body, comment.score, comment.author.name if comment.author else "Deleted",
                      datetime.utcfromtimestamp(comment.created_utc), comment.parent_id, submission.id))
            db.commit()
    except Exception as e:
        print(f"Error writing comment to PostgreSQL: {e}")


def process_submissions_and_comments(db, subreddit, search_query, limit):
    i = 1
    try:
        for submission in subreddit.search(search_query, limit=limit):
            write_posts_to_postgres(db, submission)
            insert_post_search_query(db, submission.id, search_query)
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                write_comments_to_postgres(db, comment, submission)
            print(f'{i}/{limit} finished')
            i += 1
    except Exception as e:
        print(f"Error processing submissions and comments: {e}")


def main():
    try:
        db = connect_to_postgres()
        reddit = connect_to_reddit_app()
        subreddit = reddit.subreddit('Parenting')
        write_subreddit_to_postgres(db, subreddit)
        process_submissions_and_comments(db, subreddit, 'Roblox', 1000)
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if db:
            db.close()


if __name__ == '__main__':
    main()
