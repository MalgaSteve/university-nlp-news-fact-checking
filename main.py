import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
import requests
from flask import Flask, render_template, request, jsonify
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import re


load_dotenv()
os.environ["NEWS_API_KEY"] = "3bb273e21d064447ae3844d62b155fa8"  # NewsAPI key
os.environ["FACT_CHECK_API_KEY"] = " "  # we need to put ft-check api key in ""


model = ChatOpenAI(
    model="openhermes",  # pre-trained summarization model
    base_url="http://localhost:11434/v1"
)


app = Flask(__name__)

news_aggregator = Agent(
    role="News Aggregator",
    goal="Fetch real-time news articles on selected topics using NewsAPI.",
    backstory="Responsible for gathering and providing trending news articles.",
    verbose=True,
    allow_delegation=False,
    llm=model 
)

news_aggregation_task = Task(
    description="Fetch trending news articles based on a given topic using NewsAPI.",
    expected_output="A list of relevant news articles.",
    agent=news_aggregator,
    verbose=True,
    allow_delegation=False,
    llm=model
)

def fetch_news(topic):
    api_key = "3bb273e21d064447ae3844d62b155fa8" 
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [{"title": article["title"], "content": article["content"]} for article in articles if article["content"]]
    return []

summarizer = Agent(
    role="Summarization Expert",
    goal="Summarize news articles into concise, essential summaries.",
    backstory="Skilled at identifying the key points of articles and creating brief summaries.",
    verbose=True,
    allow_delegation=False,
    llm=model
)

summarization_task = Task(
    description="Summarize the provided articles into concise summaries.",
    expected_output="Summaries of news articles with key points highlighted.",
    agent=summarizer,
    verbose=True,
    allow_delegation=False,
    llm=model 
)


def summarize_article(article_content):
    prompt = f"Summarize the following article:\n\n{article_content}\n\nSummary:"
    response = model.generate(prompt)
    return response.get("text", "") if response else "Error generating summary."

fact_checker = Agent(
    role="Fact-Checker",
    goal="Verify claims in the summaries against reliable sources.",
    backstory="Ensures accuracy by cross-referencing claims with trusted databases.",
    allow_delegation=False,
    verbose=True,
    llm = model 
)

fact_checking_task = Task(
    description="Verify claims in summaries using a fact-checking API.",
    expected_output="Fact-checked summaries with verification statuses.",
    agent=fact_checker,
    llm = model 
)

def fact_check(summary):
    api_key = os.getenv("FACT_CHECK_API_KEY")
    url = f"https://factchecktools.googleapis.com/$discovery/rest?version=v1alpha1/claims:search?key={api_key}"
    payload = {"query": summary, "pageSize": 5}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        claims = response.json().get("claims", [])
        return [{"text": claim.get("text"), "claimReview": claim.get("claimReview", [])} for claim in claims]
    return []


crew = Crew(
    agents=[news_aggregator, summarizer, fact_checker],
    tasks=[news_aggregation_task, summarization_task, fact_checking_task],
    verbose=2,
    memory=True  
)

result = crew.kickoff()


@app.route('/process', methods=['POST'])
def process():
    topic = request.json.get('topic', 'technology')
    

    news_articles = fetch_news(topic)
    summaries = [summarize_article(article['content']) for article in news_articles]
    fact_checks = [fact_check(summary) for summary in summaries]
    

    result = {
        "articles": news_articles,
        "summaries": summaries,
        "fact_checks": fact_checks
    }
    return jsonify(result)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
