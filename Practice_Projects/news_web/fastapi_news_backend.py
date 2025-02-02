from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline

app = FastAPI()

# Define news sources (RSS feeds or APIs)
NEWS_SOURCES = {
    "technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "sports": "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
}

# Load NLP Summarization Model
summarizer = pipeline("summarization")

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
    
    return articles

@app.get("/news/{category}")
def get_news(category: str):
    if category not in NEWS_SOURCES:
        return {"error": "Category not found"}
    return {"category": category, "articles": fetch_news(NEWS_SOURCES[category])}

@app.get("/")
def home():
    return {"message": "Welcome to the News API! Use /news/{category} to fetch news."}
