import flet as ft
import requests

API_URL = "http://127.0.0.1:8000/news/"

CATEGORIES = ["technology", "business", "sports"]

def fetch_news(category):
    response = requests.get(API_URL + category)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

def main(page: ft.Page):
    page.title = "News App"
    page.scroll = "adaptive"
    
    category_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(cat) for cat in CATEGORIES],
        value="technology",
        on_change=lambda e: load_news(e.control.value),
    )
    
    news_column = ft.Column()
    
    def load_news(category):
        news_column.controls.clear()
        articles = fetch_news(category)
        for article in articles:
            news_column.controls.append(
                ft.Card(
                    content=ft.Column(
                        [
                            ft.Text(article["title"], weight="bold"),
                            ft.Text(article["summary"]),
                            ft.TextButton("Read more", url=article["link"])
                        ],
                        spacing=10
                    )
                )
            )
        page.update()
    
    load_news("technology")
    
    page.add(
        ft.Column([
            ft.Text("Select News Category", size=20, weight="bold"),
            category_dropdown,
            news_column
        ], spacing=20)
    )
    
ft.app(target=main)
