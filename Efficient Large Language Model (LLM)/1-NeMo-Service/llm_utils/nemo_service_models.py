import json
import csv
import os
from typing import Callable, List, Dict

from nemollm.api import NemoLLM
from tqdm.notebook import tqdm

from .helpers import accuracy_score, sprint
from .interfaces import PromptWithExamplesCreator, TextCleaner
from .data_classes import PromptWithLabel
from .prompt_creators import create_llama_prompt_with_examples

import os
import json
import csv
from typing import List, Dict
from tqdm.notebook import tqdm

class NemoServiceBaseModel:
    """
    Base model for interacting with NVIDIA's NeMo Service for generating text and evaluating prompts.
    
    Attributes:
        model (str): The model identifier for generating text.
        create_prompt_with_examples (PromptWithExamplesCreator, optional): A function to create prompts with examples.
        customization_id (str, optional): NeMo Service customization ID to use with base model.
    """
    
    def __init__(self, model: str, create_prompt_with_examples = None, customization_id: str = None):
        """
        Initializes the NemoServiceBaseModel with necessary configurations.
        
        Parameters:
            model (str): The model identifier.
            create_prompt_with_examples (PromptWithExamplesCreator, optional): Function to create prompts.
            customization_id (str, optional): NeMo Service customization ID to use with base model.
        """
        
        self.model = model
        self.api_host = os.getenv('API_HOST')
        self.api_key = os.getenv('NGC_API_KEY')
        self.create_prompt_with_examples = create_prompt_with_examples
        self.experiment_results = {}
        self.customization_id = customization_id

    def generate(self, prompt: str, return_type:str = 'text', tokens_to_generate:int = 200, temperature:float = None, **kwargs):
        """
        Generates text based on the given prompt and parameters.
        
        Parameters:
            prompt (str): The input prompt for text generation.
            return_type (str): The type of response, either 'text' or 'stream'.
            tokens_to_generate (int): Number of tokens to generate.
            temperature (float, optional): Controls the randomness of the generation.
            
        Returns:
            str or generator: Generated text or a generator yielding text.
        """
        
        model = self.model
        customization_id = self.customization_id
        
        if return_type == 'stream':
            return self._generate_stream(prompt, model, customization_id, tokens_to_generate, temperature, **kwargs)
        else:
            return self._generate_text(prompt, model, customization_id, tokens_to_generate, temperature, **kwargs)

    def _generate_stream(self, prompt, model, customization_id, tokens_to_generate, temperature, **kwargs):
        """
        Internal method to generate text in stream mode.
        """
        conn = NemoLLM(api_host=self.api_host, api_key=self.api_key)

        response = conn.generate(
            model=model,
            customization_id=customization_id,
            prompt=prompt,
            tokens_to_generate=tokens_to_generate,
            return_type='stream',
            temperature=temperature,
            **kwargs
        )
        for raw_response in response:
            try:
                decoded_response = json.loads(raw_response.decode('utf-8'))
                sprint(decoded_response['text'])
            except json.JSONDecodeError:
                sprint("Error in decoding response.")
    
    def _generate_text(self, prompt, model, customization_id, tokens_to_generate, temperature, **kwargs):
        """
        Internal method to generate text in text mode.
        """
        conn = NemoLLM(api_host=self.api_host, api_key=self.api_key)

        response = conn.generate(
            model=model,
            prompt=prompt,
            customization_id=customization_id,
            tokens_to_generate=tokens_to_generate,
            return_type='text',
            temperature=temperature,
            **kwargs
        )
        return response

    def evaluate(self, prompts_with_labels: List[PromptWithLabel], get_clean_prediction: TextCleaner = None, print_results: bool = True, experiment_name: str = "", model_description: str = "", write_results_to_csv: bool = False, csv_file_name: str = "", **kwargs) -> float:
        """
        Evaluates the model based on a list of prompts with expected labels.
        
        Parameters:
            prompts_with_labels (List[PromptWithLabel]): A list of tuples containing prompts and expected labels.
            get_clean_prediction (TextCleaner, optional): Function to clean model predictions.
            print_results (bool): Whether to print the results.
            experiment_name (str): Name of the experiment for tracking.
            model_description (str): Description of the model.
            write_results_to_csv (bool): Whether to write the results to a CSV file.
            csv_file_name (str): The name of the CSV file to write to.
            
        Returns:
            float: The accuracy of the model predictions.
        """
        
        model = self.model
        if not model_description:
            model_description = model
        
        labels = []
        predictions = []
        
        num_correct = 0
        
        for prompt, label in tqdm(prompts_with_labels):
            prediction = self.generate(prompt, return_type='text', **kwargs)
            clean_prediction = get_clean_prediction(prediction) if get_clean_prediction else prediction
            
            labels.append(label)
            predictions.append(clean_prediction)
            
            if label == clean_prediction:
                num_correct += 1
        
        raw_accuracy = accuracy_score(labels, predictions)
        accuracy = f'{raw_accuracy:.2f}'

        if print_results:
            print(f'{num_correct}/{len(prompts_with_labels)} correct')
            print(f'Accuracy: {accuracy}')

        if experiment_name:
            self.experiment_results.setdefault(experiment_name, {}).update({model_description: accuracy})

            if write_results_to_csv:
                if not csv_file_name:
                    csv_file_name = f'{experiment_name}.csv'
                self._write_experiment_results_to_csv(csv_file_name)
            
        return accuracy

    def clear_experiment_results(self):
        """
        Clears the stored experiment results.
        """
        self.experiment_results = {}

    def _append_to_csv(self, file_name: str, experiment_name: str, data: Dict[str, str]):
        """
        Appends experiment results to a CSV file.
        """
        # Check if file exists and is not empty
        file_exists = os.path.isfile(file_name) and os.path.getsize(file_name) > 0

        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers only if the file does not exist or is empty
            if not file_exists:
                writer.writerow(['Experiment', 'Model', 'Accuracy'])

            for model, accuracy in data.items():
                writer.writerow([experiment_name, model, accuracy])

    def _write_experiment_results_to_csv(self, csv_file):
        """
        Writes all experiment results to a CSV file.
        """
        for experiment_name, model_results in self.experiment_results.items():
            self._append_to_csv(csv_file, experiment_name, model_results)


class LlamaChatbot:
    """
    A chatbot interface for generating conversational responses using the NemoServiceBaseModel,
    specifically for LLaMA models.
    """

    def __init__(self, nemo_service_llama_model, system_context):
        """
        Initializes a new instance of the LlamaChatbot class.

        Parameters:
        - nemo_service_llama_model (NemoServiceBaseModel): An instance of NemoServiceBaseModel to handle response generation.
        - system_context (str): A string that sets the initial context for the language model.
        """
        self.nemo_service_llama_model = nemo_service_llama_model
        self.system_context = system_context
        self.conversation_history = []  # Initializes the conversation history
        self.create_prompt_with_examples = create_llama_prompt_with_examples

    def chat(self, user_msg):
        """
        Generates a response from the chatbot based on the user's message.

        This method constructs a prompt with the current system context and conversation history,
        sends it to the NemoServiceBaseModel, and then stores the new user message and model's response
        in the conversation history.

        Parameters:
        - user_msg (str): The user's message to which the chatbot will respond.

        Returns:
        - str: The generated response from the chatbot.
        """
        prompt = self._construct_prompt(user_msg)
        agent_response = self.nemo_service_llama_model.generate(prompt).strip()

        # Store this interaction in the conversation history
        self.conversation_history.append((user_msg, agent_response))

        return agent_response

    def _construct_prompt(self, user_msg):
        """
        Constructs a prompt for the language model incorporating the system context, 
        conversation history, and the latest user message using the provided
        'create_llama_prompt_with_examples' helper function.

        Parameters:
        - user_msg (str): The latest message from the user.

        Returns:
        - str: The constructed prompt.
        """
        # Use the create_prompt_with_examples method if available
        return self.create_prompt_with_examples(
            main_prompt=user_msg,
            conversation_examples=self.conversation_history,
            system_context=self.system_context
        )

    def reset(self):
        """
        Resets the conversation history of the chatbot.
        """
        self.conversation_history = []
