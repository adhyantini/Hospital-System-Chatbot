# fine_tuning.py

import json
import openai
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load fine-tuning data from a JSON file
# This JSON file should contain a list of dictionaries with 'prompt' and 'response' keys.
with open('fine_tuning_data.json', 'r') as file:
    fine_tuning_data = json.load(file)

# Function to prepare training data by converting it into the required format
def prepare_training_data(data):
    training_examples = []
    for example in data:
        # Each example is a dictionary with 'prompt' and 'completion' keys, used for fine-tuning
        training_examples.append({"prompt": example["prompt"], "completion": example["response"]})
    return training_examples

# Prepare the training data from the loaded JSON file
training_data = prepare_training_data(fine_tuning_data)

# Function to fine-tune the model using the provided training data
def fine_tune_model(training_data):
    # Define the template for the prompt
    # This template includes a context that the model is a hospital management assistant
    template = """
    You are a knowledgeable assistant specializing in hospital management.
    Please answer the following questions accurately and with the context provided.
    
    Question: {question}
    Answer:
    """
    
    # Create a PromptTemplate object with the input variable 'question' and the defined template
    prompt = PromptTemplate(input_variables=["question"], template=template)

    # Initialize the OpenAI language model (LLM) with the specified model name
    llm = OpenAI(model_name="text-davinci-003")
    
    # Initialize conversation memory to keep track of the conversation context
    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    
    # Create an LLMChain with the language model, prompt template, and memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # Loop through the training data and generate fine-tuned responses
    for example in training_data:
        # Run the LLMChain with the provided question and get the generated response
        response = chain.run(question=example["prompt"])
        # Print the fine-tuned response for each prompt
        print(f"Fine-tuned response: {response}")

    # Indicate that the fine-tuning process is completed
    print("Fine-tuning completed.")

# Main entry point of the script
if __name__ == "__main__":
    # Call the fine-tuning function with the prepared training data
    fine_tune_model(training_data)
