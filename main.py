# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from typing import ClassVar
import requests
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from tools import FetchNews, FactCheckTool
from flask import Flask, request, jsonify, render_template
import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"


app = Flask(__name__) 

llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    verbose=True
)


news_aggregator = Agent(
        role='News Aggregator',
        goal='Research in real-time the most relevant news',
        backstory='You are an assistant specialized finding the most relevant news about {topic}',
        verbose=True,
        allow_delegation=False,
        tool=[FetchNews],
        llm=llm)

summerizer = Agent(
        role='News Summerizer',
        goal='Write comppelling and engeaging blog posts about {topic}',
        backstory='You are an expert at give conscice and clear summerizations of news articles about {topic}',
        verbose=True,
        allow_delegation=False,
        llm=llm)

fact_checker = Agent(
        role='Fact Checker',
        goal='cross-reference key claims in the summaries with existing fact-checked sources or reliable news databases',
        backstory='An expert at {topic} which has a lot of experience in fact-checking news articles of {topic}',
        verbose=True,
        allow_delegation=False,
        tool=[FactCheckTool],
        llm=llm,
    )

task1 = Task(description='Gather in real-time the most relevant news', 
             agent=news_aggregator,
             expected_output='A list with the most relevant articles about {topic}',
             )
task2 = Task(description='Summarize news articles about {topic}',
             agent=summerizer,
             expected_output='A summarization of news about {topic}')

task3 = Task(description='Cross reference key claims in the summaries with exisiting fact checking sources',
             agent=fact_checker,
             expected_output='Verify claims within summaries, marking each as Verified, Partially Verified, or Unverified based on reliable cross-referencing in html format',
             )


crew = Crew(
        agents = [news_aggregator, summerizer, fact_checker],
        tasks = [task1, task2, task3],
        verbose=True,
        process = Process.sequential
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_topic():
    topic = request.form.get("topic")

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        # Run the process using crew
        result = crew.kickoff(inputs={"topic": topic})
        print(result.raw)
        print(task3.output.raw)

        # Ensure result is properly formatted (if crew returns complex data)
        if isinstance(result.raw, dict):
            return jsonify(result.raw), 200
        elif isinstance(task3.output.raw, str):
            return task3.output.raw, 200
        else:
            return jsonify({"error": "Unexpected result format"}), 500

    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error processing topic: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
