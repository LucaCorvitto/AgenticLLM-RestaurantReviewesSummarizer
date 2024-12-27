# Agentic LLM - Restaurant Reviewes Summarizer
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![AutoGen](https://img.shields.io/badge/Autogen-Framework-green?logo=powerautomate&logoColor=black)

This project has been done as part of the [Large Language Model Agents MOOC](https://llmagents-learning.org/f24). All the files can be downloaded from their website to replicate the assignment from scratch for educational purpose.

## Why LLMs
Large Language Models (LLMs) are extremely flexible tools and are effective in tasks that require understanding human language. Their pretraining procedure, which involves next token prediction on vast amounts of text data, enables them to develop an understanding of syntax, grammar, and linguistic meaning. One strong use case is the ability to analyze large amounts of unstructured text data, allowing us to derive insights more efficiently.

## AutoGen Framework
[AG2](https://ag2.ai/), formerly AutoGen, is a framework that enables the creation of multi-agent workflows involving multiple LLMs. Essentially, is a way to define "control flows" or "conversation flows" between multiple LLMs. This way, one can chain together several individual LLM agents, having them all work together by conversing with each other to accomplish a larger task. Through AutoGen, users can define networks of LLM agents, enabling complex reasoning, self-evaluating, data processing pipelines, and much more.

## Task Description
![image](https://github.com/user-attachments/assets/53cc1c34-d8ad-46f7-92a1-372bda7194cf)

The above figure is a diagram of the architecture followed by this project. It follows a sequential conversation pattern between two agents. The pipeline is essentially a directed graph, first fetching the restaurant reviews, analyzing them, then calling a function, but with an additional "supervising" entrypoint agent.

The entry point agent is always the agent responsible for initiating the chat with other agents. Relevant summaries of previous chats between agent pairs are carried over as contexts to other chats.

### Task 1: Fetching the Relevant Data
The first step is to figure out which restaurant review data is needed.
The data fetch agent is used to analyze the query to determine the correct function call to the fetch function. This data fetch agent will suggest a function call with particular arguments. Then, the entry point agent will execute the suggested function call.

### Task 2: Analyzing Reviews
The next step is performed by an agent that analyzes the reviews fetched in the previous section. More specifically, this agent look at every single review corresponding to the queried restaurant and extract two scores:
- `food_score`: the quality of food at the restaurant. This will be a score from 1-5. 
- `customer_service_score`: the quality of customer service at the restaurant. This will be a score from 1-5. 

The agent, actually, extracts these two scores by looking for keywords in the review. Each review has keyword adjectives that correspond to the score that the restaurant should get for its `food_score` and `customer_service_score`. In this case the keywords are set and predefined, but one can change the prompt of the agent in order to handle unseen adjectives since the model is a LLM.

### Task 3: Scoring Agent
The final step is done using a scoring agent. This agent looks at all of the review's `food_score` and `customer_service_score` to make a final function call to `calculate_overall_score`. At the end of this section the overall system is able to answer queries regarding any restaurant in the list of restaurants.

## Collaborators
[Gabriele Corvitto](https://github.com/Gabriollo94)
