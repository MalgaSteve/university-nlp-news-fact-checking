from typing import ClassVar
import requests
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import os
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FACT_API_KEY = os.getenv("FACT_API_KEY")

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
                return f"No news articles found for topic: {topic}"

            # Summarize the top articles
            articles = data["articles"]
            result = f"Top {len(articles)} news articles for '{topic}':\n\n"
            for i, article in enumerate(articles, 1):
                result += (
                    f"{i}. **{article['title']}**\n"
                    f"   Source: {article['source']['name']}\n"
                    f"   Link: {article['url']}\n"
                    f"   Description: {article['description'] or 'No description available.'}\n\n"
                    f"   Content: {article['content'] or 'No content available.'}\n\n"
                    )
            return result

        except requests.exceptions.RequestException as e:
            return f"Error fetching news: {str(e)}"

class FactCheckTool(BaseTool):
    name: str = "FactCheckTool"
    description: str = "Cross-references claims with fact-check data from reliable sources."

    def __init__(self):
        self.api_key = FACT_API_KEY
        self.api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def _run(self, claim: str) -> str:
        # Construct the API request URL
        params = {
            'query': claim,
            'key': self.api_key,
        }

        # Make the request to Google Fact Check API
        response = requests.get(self.api_url, params=params)

        if response.status_code != 200:
            return f"Error: Unable to fetch data from the API. Status Code: {response.status_code}"
        
        # Parse the JSON response
        data = response.json()
        
        # If no fact-checks found, return a message
        if 'claims' not in data or not data['claims']:
            return f"No fact-checks found for the claim: '{claim}'."
        
        # Extract relevant information
        fact_check_data = []
        for claim_data in data['claims']:
            claim_text = claim_data['text']
            claim_date = claim_data['claimDate']
            claim_url = claim_data['claimReview'][0]['url']
            source_name = claim_data['claimReview'][0]['publisher']['name']
            
            fact_check_data.append({
                'claim': claim_text,
                'date': claim_date,
                'source': source_name,
                'url': claim_url,
            })

        # Return a formatted message with the fact-check results
        result = f"Fact-check data for the claim: '{claim}':\n"
        for idx, fact in enumerate(fact_check_data, 1):
            result += f"\n{idx}. Claim: {fact['claim']}\nDate: {fact['date']}\nSource: {fact['source']}\nURL: {fact['url']}\n"

        return result
