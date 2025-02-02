from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
import sqlite3

app = FastAPI()

# Define news sources (RSS feeds or APIs)
NEWS_SOURCES = {
    "technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "sports": "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
}

NEWSAPI_KEY = "your_newsapi_key_here"

# Load NLP Summarization Model
summarizer = pipeline("summarization")

# Initialize SQLite database
conn = sqlite3.connect("news.db")
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        title TEXT,
        summary TEXT,
        link TEXT
    )
""")
conn.commit()

def fetch_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    articles = []
    
    for item in soup.find_all("item")[:5]:  # Limit to 5 articles per category
        link = item.find("link").text
        title = item.find("title").text
        article = Article(link)
        article.download()
        article.parse()
        summary = summarizer(article.text[:1000], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        
        articles.append({"title": title, "summary": summary, "link": link})
        
        c.execute("INSERT INTO news (category, title, summary, link) VALUES (?, ?, ?, ?)", (url, title, summary, link))
        conn.commit()
    
    return articles

def fetch_newsapi(category):
    url = f"https://newsapi.org/v2/top-headlines?category={category}&apiKey={NEWSAPI_KEY}"
    response = requests.get(url).json()
    articles = []
    
    for article in response.get("articles", [])[:5]:
        title = article["title"]
        link = article["url"]
        summary = summarizer(article["description"][:1000], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        
        articles.append({"title": title, "summary": summary, "link": link})
        
        c.execute("INSERT INTO news (category, title, summary, link) VALUES (?, ?, ?, ?)", (category, title, summary, link))
        conn.commit()
    
    return articles

@app.get("/news/{category}")
def get_news(category: str):
    if category in NEWS_SOURCES:
        return {"category": category, "articles": fetch_news(NEWS_SOURCES[category])}
    else:
        return {"category": category, "articles": fetch_newsapi(category)}

@app.get("/")
def home():
    return {"message": "Welcome to the News API! Use /news/{category} to fetch news."}

# Implement a Flutter frontend using Flet (to be developed separately)
