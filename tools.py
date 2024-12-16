from typing import ClassVar
import requests
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import os
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class FetchNews(BaseTool):
    name: str = "Fetch news using NewsAPI"
    description: str = "This tool fetches news using NewsAPI"
    API_KEY: ClassVar[str] = NEWS_API_KEY
    BASE_URL: ClassVar[str] = "https://newsapi.org/v2/everything"

    def _run(self, topic:str) -> str:
        try:
            params = {
                "q": topic,
                "apiKey": self.API_KEY,
                "pageSize": 5,  # Limit the number of results
                "sortBy": "relevance"
            }
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Parse the response JSON
            data = response.json()

            # Handle empty or missing results
            if data.get("status") != "ok" or not data.get("articles"):
                return f"No news articles found for query: {query}"

            # Summarize the top articles
            articles = data["articles"]
            result = f"Top {len(articles)} news articles for '{query}':\n\n"
            for i, article in enumerate(articles, 1):
                result += (
                    f"{i}. **{article['title']}**\n"
                    f"   Source: {article['source']['name']}\n"
                    f"   Link: {article['url']}\n"
                    f"   Description: {article['description'] or 'No description available.'}\n\n"
            ***REMOVED***
            return result

        except requests.exceptions.RequestException as e:
            return f"Error fetching news: {str(e)}"
