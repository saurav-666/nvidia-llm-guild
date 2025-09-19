import json
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

import ipywidgets as widgets
from IPython.display import display

def edit_list(input_list):
    index = 0  # Use a simple integer for the index

    def on_button_clicked(b):
        nonlocal index  # Declare index as nonlocal to modify it
        if b.description == 'Remove':
            input_list.pop(index)
        elif b.description == 'Replace':
            input_list[index] = text_input.value  # Replace with the text input value
        else:
            index += 1

        if index < len(input_list):
            label.value = f"Do you want to keep '{input_list[index]}' in the list?"
            text_input.value = ''  # Clear the text input
        else:
            # Correctly update label value and clear the widgets
            label.value = "Review complete."
            btn_keep.layout.display = 'none'
            btn_remove.layout.display = 'none'
            btn_replace.layout.display = 'none'
            text_input.layout.display = 'none'

    if input_list:
        label = widgets.Label(f"Do you want to keep '{input_list[index]}' in the list?")
        btn_keep = widgets.Button(description="Keep")
        btn_remove = widgets.Button(description="Remove")
        btn_replace = widgets.Button(description="Replace")
        text_input = widgets.Text(placeholder="Enter replacement")

        btn_keep.on_click(on_button_clicked)
        btn_remove.on_click(on_button_clicked)
        btn_replace.on_click(on_button_clicked)

        display(label, widgets.HBox([btn_keep, btn_remove, btn_replace]), text_input)
    else:
        print("The list is empty.")


def plot_experiment_results(csv_file):
    # Ignore printing seaborn warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)  # Ignore FutureWarnings

        data = pd.read_csv(csv_file)

        data['Accuracy'] = pd.to_numeric(data['Accuracy'])

        # Setting the plot style and color palette
        sns.set(style="whitegrid")
        sns.set_palette("pastel")

        # Plot for Accuracy
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Accuracy', hue='Experiment', data=data)
        plt.title('Accuracy by Model and Experiment')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.show()


def sprint(generator):
    for text in generator:
        # Print the text without adding a newline, unless the text itself ends with a newline
        print(text, end="" if not text.endswith('\n') else "\n")

def accuracy_score(labels, predictions):
    correct_predictions = 0
    total_predictions = len(predictions)

    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def collect_my_prompts_and_responses():
    def handle_submission(sender):
        data = {"prompt": prompt_textarea.value, "completion": response_textarea.value}
        with open('my_prompts_and_responses.jsonl', 'a') as file:
            file.write(json.dumps(data) + "\n")
        prompt_textarea.value = ''
        response_textarea.value = ''
        update_sample_count()

    def update_sample_count():
        try:
            with open('my_prompts_and_responses.jsonl', 'r') as file:
                count = sum(1 for _ in file)
        except FileNotFoundError:
            count = 0
        num_samples_label.value = f'Number of Samples: {count}'

    prompt_textarea = widgets.Textarea(description='Prompt:', layout={'width': '100%', 'height': '100px'})
    response_textarea = widgets.Textarea(description='Response:', layout={'width': '100%', 'height': '100px'})
    submit_button = widgets.Button(description="Submit")
    num_samples_label = widgets.Label()

    submit_button.on_click(handle_submission)

    container = widgets.VBox([widgets.Label(value='My Prompts and Responses'),
                              prompt_textarea, 
                              response_textarea, 
                              submit_button, 
                              num_samples_label],
                             layout=widgets.Layout(overflow='hidden'))

    display(container)
    update_sample_count()