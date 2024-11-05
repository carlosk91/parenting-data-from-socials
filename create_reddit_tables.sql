CREATE TABLE reddit_posts (
    reddit_post_id TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    score INTEGER,
    author TEXT,
	subreddit_id VARCHAR(255) UNIQUE,
    timestamp TIMESTAMP,
	created_at TIMESTAMP DEFAULT current_timestamp,
	updated_at TIMESTAMP DEFAULT current_timestamp
);


CREATE TABLE reddit_comments (
    reddit_comment_id TEXT PRIMARY KEY,
    body TEXT,
    score INTEGER,
    author TEXT,
    timestamp TIMESTAMP,
    parent_id TEXT,
    reddit_post_id TEXT,
	created_at TIMESTAMP DEFAULT current_timestamp,
	updated_at TIMESTAMP DEFAULT current_timestamp
);


CREATE TABLE reddit_subreddits (
    subreddit_id VARCHAR(255) UNIQUE,
    display_name VARCHAR(255) UNIQUE,
    subscriber_count INT,
    public_description TEXT,
    creation_date TIMESTAMP,
    active_user_count INT,
	created_at TIMESTAMP DEFAULT current_timestamp,
	updated_at TIMESTAMP DEFAULT current_timestamp
);


CREATE TABLE reddit_post_search_queries (
    reddit_post_id VARCHAR(255),
    search_query VARCHAR(255),
    created_at TIMESTAMP DEFAULT current_timestamp,
	updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (reddit_post_id, search_query)
);