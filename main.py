# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from typing import ClassVar
import requests
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from tools import FetchNews, FactCheckTool

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
        backstory='You are an expert at summerizer news articles about {topic}',
        verbose=True,
        allow_delegation=False,
        llm=llm)

fact_checker = Agent(
        role='Fact Checker',
        goal='cross-reference key claims in the summaries with existing fact-checked sources or reliable news databases',
        backstory='An expert at {topic} which has an expertise in fact-checking news articles',
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
             expected_output='A summarization of news about {topic} in markdown format')

task3 = Task(description='Cross reference key claims in the summaries with exisiting fact checking sources',
             agent=fact_checker,
             expected_output='A markdown file which includes each summarization and an overview if they are real or fake news',
             output_file='result.md')


crew = Crew(
        agents = [news_aggregator, summerizer, fact_checker],
        tasks = [task1, task2, task3],
        verbose=True,
        process = Process.sequential
        )

topic = "Trump"

test_tool = FetchNews()
test_run, out = test_tool.run(topic)
print(test_run, " Out:\n", out)

test_tool_check = FactCheckTool()
print(test_tool_check.run(out))

result = crew.kickoff(inputs={"topic": topic})
print(result)
