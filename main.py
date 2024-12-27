from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import numpy as np

# GLOBAL VARIABLES DEFINITION
RES_DICT : dict [str, List[str]] = {}
REVIEWS : list[str] = []

# FUNCTIONS DEFINITION
def create_restaurant_dict():
    res_dict = dict()
    with open('restaurant-data.txt','r') as f:
        for line in f:
            list_line = line.split('.')
            rest_name = list_line[0]
            review = '.'.join(list_line[1:]).strip()
            if rest_name not in res_dict:
                reviews = [review]
                res_dict[rest_name] = reviews
            else:
                res_dict[rest_name].append(review)

    return res_dict


def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    global REVIEWS
    REVIEWS = RES_DICT[restaurant_name]

    return {restaurant_name: REVIEWS}


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    N = min(len(food_scores), len(customer_service_scores))
    SUM = 0
    for i in range(N):
        SUM += np.sqrt(food_scores[i]**2 * customer_service_scores[i])
    
    output = (SUM / (N * np.sqrt(125))) * 10

    output_str = "{:.4f}".format(output)

    return {restaurant_name : output_str}

def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    system_prompt =f"""
            You are the Data Fetch Agent responsible for retrieving reviews for a specific restaurant.
            You have to extract the restaurant name from the following user query:
            {restaurant_query}
            Then, suggest the extracted restaurant name as argument to the function `fetch_restaurant_data`, that will be executed by the entrypoint agent.

            Steps:
            1. Extract the restaurant name from the user query.
                Keep in mind that the restaurants names, properly formatted, are: {RES_DICT.keys()}
                Take into account any possible spelling mistake and assign the query to only one of these possible restaurants.
            2. Call the function `fetch_restaurant_data(restaurant_name)`.

            Example:
            - User query: "How good is the food at Subway?"
            - Output: fetch_restaurant_data("Subway")
    """
    return system_prompt

def main(user_query: str):

    global RES_DICT
    RES_DICT = create_restaurant_dict()

    # SYSTEM MESSAGES DEFINITION
    entrypoint_agent_system_message = """
            You are the Entry Point Agent responsible for understanding user queries and coordinating with other agents.
            Given a user query, your task is divided in different steps:
            
            1. Trigger the Data Fetch Agent.
            2. Receive the Data Fetch Agent output and call the `fetch_restaurant_data()` function using the suggested argument.
            3. Collect the function output and pass it to the Data Analysis Agent.
            4. Receive the Data Analysis Agent output and pass it to the Scoring Agent.
            5. Receive the Scoring Agent output and execute `calculate_overall_score()` with the suggested arguments.
            6. Write an answer to the user based on the function's output.

            Example:
            - User query: "How good is the food at Subway?"
            1. Trigger Data Fetch Agent.
            2. Given the Data Fetch Agent suggested function call with argument "Subway", execute `fetch_restaurant_data("Subway")`.
            3. Collect the list of reviews from the function output and pass them as they are to the Data Analysis Agent.
            4. Given the Data Analysis Agent output "[food_score:[4, 5, 3], customer_service_score: [5, 4, 4]]" pass it to the Scoring Agent.
            5. Given the Scoring Agent suggested function call with argument "Subway",[4, 5, 3],[5, 4, 4], execute `calculate_overall_score("Subway",[4, 5, 3],[5, 4, 4])`.
            6. Collect the "5.035" function output and generate an answer: "The average food quality score is 5.035. 
            
            Focus on working with the other agent accordingly to accomplish your task. 
    """

    data_fetch_agent_system_message = get_data_fetch_agent_prompt(user_query)

    review_analysis_agent_system_message = """
            You are an expert review analyzer designed to extract numerical scores from restaurant reviews based on specific keyword adjectives. Your task is to analyze each review provided, identify the keywords describing the food and customer service, and output two scores:

            **food_score**: The quality of the food at the restaurant, rated on a scale from 1 to 5.
            **customer_service_score**: The quality of the customer service at the restaurant, rated on a scale from 1 to 5.
            
            Instructions:
            Each review contains exactly two keywords (one describing the food and one describing the customer service). These keywords correspond to the scores listed below:
            
            Score 1/5: Awful, horrible, disgusting.
            Score 2/5: Bad, unpleasant, offensive.
            Score 3/5: Average, uninspiring, forgettable.
            Score 4/5: Good, enjoyable, satisfying.
            Score 5/5: Awesome, incredible, amazing.
            
            Match the keyword describing the food to its corresponding score and assign it to food_score.
            Match the keyword describing the customer service to its corresponding score and assign it to customer_service_score.
            Ignore all other text in the review; only the keyword adjectives listed above determine the scores.
            
            Example:
            Review:
            The food at McDonald's was average, but the customer service was unpleasant. The uninspiring menu options were served quickly, but the staff seemed disinterested and unhelpful.

            Analysis:
            Food is described as "average" -> food_score: 3.
            Customer service is described as "unpleasant" -> customer_service_score: 2.
            Output:
            food_score: 3, customer_service_score: 2

            Output Format:
            For every review, provide the extracted scores in this format:

            [food_score:<Score>, customer_service_score: <Score>]

            Begin analyzing the reviews now.
    """
    scoring_agent_system_message = """ 
            You are the Scoring Agent responsible for calculating the overall score for a restaurant.
            Use the scores provided by the Entrypoint Agent to calculate the average food quality and service quality.

            Steps:
            1. Take the list of food and service scores.
            3. Provide the three needed argument for the function call, that are: restaurant_name, food_service_score, customer_service_score.
            2. Compute the average score for food and customer service calling the function `calculate_overall_score(restaurant_name, food_scores, customer_service_scores)`, ensuring the correct number of arguments.


            Example Input:
            - [food_score: 1, customer_service_score: 1]
            - [food_score: 2, customer_service_score: 2]
            - [food_score: 3, customer_service_score: 3]
            - [food_score: 4, customer_service_score: 4]
            - [food_score: 5, customer_service_score: 5]

            Arguments: 
            {"restaurant_name":"Taco Bell","food_scores":[1,2,3,4,5],"customer_service_scores":[1,2,3,4,5]}
            
            Output:
            - 5.035
    """

    # LLM config for the entrypoint agent
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    # main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                        system_message=entrypoint_agent_system_message, 
                                        llm_config=llm_config)
    # other agents.
    data_fetch_agent = ConversableAgent("data_fetch_agent",
                                        system_message=data_fetch_agent_system_message,
                                        description="Determines relevant data to fetch, suggesting arguments for fetch_restaurant_data",
                                        llm_config=llm_config,
                                        human_input_mode="NEVER",)

    review_analysis_agent = ConversableAgent("review_analysis_agent",
                                        system_message=review_analysis_agent_system_message,
                                        description="Analyzes the unstructured, text restaurant reviews.",
                                        llm_config=llm_config,
                                        human_input_mode="NEVER",)

    scoring_agent = ConversableAgent("scoring_agent",
                                        system_message=scoring_agent_system_message,
                                        description="Suggests function call to calculate_overall_score with arguments.",
                                        llm_config=llm_config,
                                        human_input_mode="NEVER",)

    # register the functions within each agent
    entrypoint_agent.register_for_llm(name="fetch_restaurant_data", description="Fetches the reviews for a specific restaurant.")(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)

    entrypoint_agent.register_for_llm(name="calculate_overall_score", description="Compute the overall score for the associated restaurant.")(calculate_overall_score)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)

    data_fetch_agent.register_for_llm(name="fetch_restaurant_data", description="Fetches the reviews for a specific restaurant.")(fetch_restaurant_data)
    scoring_agent.register_for_llm(name="calculate_overall_score", description="Compute the overall score for the associated restaurant.")(calculate_overall_score)

    ### sequential chat
    fetch_data = entrypoint_agent.initiate_chat(
            recipient= data_fetch_agent,
            message= user_query,
            max_turns= 2,
            summary_method="reflection_with_llm",
    )

    # extract context to pass over
    beautyfied_reviews = "".join(f"{i}. {review}\n" for i, review in enumerate(REVIEWS))
    context = f"{fetch_data.summary} \n {beautyfied_reviews}"

    review_analysis = entrypoint_agent.initiate_chat(
            recipient= review_analysis_agent,
            message= context,
            max_turns= 1,
            summary_method="last_msg",
    )

    # extract context to pass over
    scores = review_analysis.summary

    scoring_result = entrypoint_agent.initiate_chat(
            recipient= scoring_agent,
            message= scores,
            max_turns= 2,
            summary_method="last_msg",
    )

    final_score = scoring_result.summary

    return final_score


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])