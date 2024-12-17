# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import BaseTool
from typing import List
import os

class FileAccessTool(BaseTool):
    name: str = "file_reader"
    description: str = "Reads the content of a file given its path. Useful for understanding file structure or analyzing content."

    def _run(self, file_path: str) -> str:
        """
        Reads the content of the provided file paths.

        Args:
            file_paths (List[str]): A list of file paths to read.
        
        Returns:
            str: The content of the files.
        """
        content = []
        if os.path.isfile(file_path):
           try:
               with open(file_path, "r", encoding="utf-8") as f:
                   file_content = f.read()
                   content.append(f"Content of {file_path}:\n{file_content}\n")
           except Exception as e:
               content.append(f"Error reading {file_path}: {str(e)}")
        else:
           content.append(f"File not found: {file_path}")
        
        return "\n".join(content)

class FileWriteTool(BaseTool):
    name: str = "file_writer"
    description: str = "Creates or overwrites a file with specified content. Useful for saving generated code or data."

    def _run(self, file_path: str, content: str) -> str:
        """
        Creates or overwrites a file with the provided content.

        Args:
            file_path (str): The path of the file to write to.
            content (str): The content to be written to the file.

        Returns:
            str: A message indicating success or failure.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            return f"File '{file_path}' has been successfully created/overwritten."
        except Exception as e:
            return f"An error occurred while writing to the file '{file_path}': {str(e)}"

tool = FileAccessTool()
file_paths = "../main.py"
result = tool._run(file_paths)
print(result)

file_tool = FileWriteTool()
result = file_tool._run("test_output.txt", "Hello, this is a test file created by FileWriteTool!")
print(result)

access_tool = FileAccessTool()
write_tool = FileWriteTool()

llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    verbose=True
)


code_analyzer = Agent(
    role="Python Code Analyzer",
    goal="Analyze Python code structure and functionality",
    backstory="Senior python developer with deep understanding of code patterns and best practices.",
    allow_code_execution=True,
    code_execution_mode="unsafe",
    allow_delegation=False,
    verbose=True,
    tools=[access_tool],
    llm=llm
)

commenter = Agent(
    role="Code Commenter",
    goal="Add clear and concise comments to Python code",
    backstory="Senior developer skilled in writing explanatory comments for code readability.",
    allow_code_execution=True,
    code_execution_mode="unsafe",
    allow_delegation=False,
    verbose=True,
    tools=[access_tool, write_tool],
    llm=llm
)

documentor = Agent(
    role="Documentation Generator",
    goal="Generate comprehensive documentation for Python code",
    backstory="Senior Technical writer specialized in creating clear and informative code documentation.",
    allow_code_execution=True,
    code_execution_mode="unsafe",
    allow_delegation=False,
    verbose=True,
    tools=[write_tool],
    llm=llm
)

analysis_task = Task(
    description="Analyze the following Python code files:\n\n {code_file_1} and provide a summary of its structure and functionality.",
    agent=code_analyzer,
    expected_output="A summary of the structure and functionality of the files"
)

comment_task = Task(
    description="Add appropriate comments to the analyzed Python code to improve readability.",
    agent=commenter,
    expected_output="two modified python file with added comments and improved readability",
    output_file="./gen_main.py"
)

documentation_task = Task(
    description="Generate comprehensive documentation for the commented Python code, including function descriptions and usage examples.",
    agent=documentor,
    expected_output="A readable and conscise documentation in markdown format",
    output_file="doc.md"
)

code_improvement_crew = Crew(
    agents=[code_analyzer, commenter, documentor],
    tasks=[analysis_task, comment_task, documentation_task]
)

inputs = {
        "code_file_1": "../main.py",
               }

result = code_improvement_crew.kickoff(inputs=inputs)
print(result)
