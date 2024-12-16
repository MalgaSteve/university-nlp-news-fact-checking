# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from typing import ClassVar
import requests
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from tools import FetchNews

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
        llm=llm
    ***REMOVED***

write = Agent(
        role='Writer',
        goal='Write comppelling and engeaging blog posts about {topic}',
        backstory='You are an expert at summerizer news articles about {topic}',
        verbose=True,
        allow_delegation=False,
        llm=llm
    ***REMOVED***

task1 = Task(description='Gather in real-time the most relevant news', 
             agent=news_aggregator,
             expected_output='A list with the most relevant articles about {topic}',
         ***REMOVED***
task2 = Task(description='Summarize news articles about {topic}',
             agent=write,
             expected_output='A summarization of news about {topic} in markdown format')

crew = Crew(
        agents = [news_aggregator, write],
        tasks = [task1, task2],
        verbose=True,
        process = Process.sequential
    ***REMOVED***

topic = "Trump"

result = crew.kickoff(inputs={"topic": topic})
print(result)
