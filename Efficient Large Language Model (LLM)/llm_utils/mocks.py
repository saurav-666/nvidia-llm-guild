import os
import datetime
import uuid

def upload(file_path):
    # Generating a random UUID
    random_id = str(uuid.uuid4())

    # Getting the size of the file in bytes
    file_size = os.path.getsize(file_path)

    # Counting the number of lines in the file
    with open(file_path, 'r') as file:
        number_of_samples = sum(1 for _ in file)

    # Current time in the specified format
    current_time = datetime.datetime.now().isoformat() + 'Z'

    # Constructing the dictionary with the random UUID
    file_info = {
        'id': random_id,
        'name': file_path,
        'size': file_size,
        'number_of_samples': number_of_samples,
        'format': 'jsonl',
        'usage_category': 'dataset',
        'org_id': 'abcdefghijkl',
        'user_id': 'abcdefghijklmnopqrstuvwxyz',
        'ready_at': '0001-01-01T00:00:00Z',
        'created_at': current_time
    }

    return file_info


def create_pubmedqa_lora_customization(**kwargs):
    def _is_valid_model(model):
        return model == 'gpt-8b-000-lora'

    def _is_valid_train_data(data_id):
        return data_id == 'cb1aab08-e396-41a8-9334-571c6672033d'

    def _is_valid_val_data(data_id):
        return data_id == '42d75e3a-7aa9-46fa-b1c0-63d7a66f7a8f'

    valid_kwargs = {'model', 'name', 'description', 'batch_size', 
                    'adapter_dim', 'training_dataset_file_id',
                    'validation_dataset_file_id', 'epochs'}

    for kwarg in kwargs.keys():
        if kwarg not in valid_kwargs:
            return f'{kwarg} is not a valid argument.'


    if 'model' not in kwargs.keys():
        return '`model` parameter not provided.'

    model = kwargs['model']
    if not _is_valid_model(model):
        return f'{model} is not the correct `model`.'


    if 'training_dataset_file_id' not in kwargs.keys():
        return '`training_dataset_file_id` parameter not provided.'

    train_data_id = kwargs['training_dataset_file_id']
    if not _is_valid_train_data(train_data_id):
        return f'{train_data_id} is not the correct `training_dataset_file_id`.'


    if 'validation_dataset_file_id' not in kwargs.keys():
        return '`validation_dataset_file_id` parameter not provided.'

    val_data_id = kwargs['validation_dataset_file_id']
    if not _is_valid_val_data(val_data_id):
        return f'{val_data_id} is not the correct `validation_dataset_file_id`.'


    if 'adapter_dim' not in kwargs.keys():
        return '`adapter_dim` parameter not provided.'

    adapter_dim = kwargs['adapter_dim']
    if adapter_dim not in {8, 12, 16, 32, 64}:
        return f'{adapter_dim} is not a valid `adapter_dim`.'


    if 'epochs' not in kwargs.keys():
        return '`epochs` parameter not provided.'

    epochs = kwargs['epochs']
    if not isinstance(epochs, int) or epochs < 1 or epochs > 50:
        return f'{epochs} is not a valid value for  `epochs`.'


    customization_id = 'cab8b23d-b49d-4e35-bfad-3abc572d8f09'
    return f'LoRA customization job for GPT8B succesfully launched! Customization ID: {customization_id}'

def create_list_gen_lora_customization(**kwargs):
    def _is_valid_model(model):
        return model == 'gpt-8b-000-lora'

    def _is_valid_train_data(data_id):
        return data_id == '85218a48-86a5-46d8-94cf-96a24f3078fa'

    def _is_valid_val_data(data_id):
        return data_id == '419c55e3-2fbc-41cb-9bed-c0482f3ba26d'

    valid_kwargs = {'model', 'name', 'description', 'batch_size',
                    'adapter_dim', 'training_dataset_file_id',
                    'validation_dataset_file_id', 'epochs'}

    for kwarg in kwargs.keys():
        if kwarg not in valid_kwargs:
            return f'{kwarg} is not a valid argument.'


    if 'model' not in kwargs.keys():
        return '`model` parameter not provided.'

    model = kwargs['model']
    if not _is_valid_model(model):
        return f'{model} is not the correct `model`.'


    if 'training_dataset_file_id' not in kwargs.keys():
        return '`training_dataset_file_id` parameter not provided.'

    train_data_id = kwargs['training_dataset_file_id']
    if not _is_valid_train_data(train_data_id):
        return f'{train_data_id} is not the correct `training_dataset_file_id`.'


    if 'validation_dataset_file_id' not in kwargs.keys():
        return '`validation_dataset_file_id` parameter not provided.'

    val_data_id = kwargs['validation_dataset_file_id']
    if not _is_valid_val_data(val_data_id):
        return f'{val_data_id} is not the correct `validation_dataset_file_id`.'


    if 'adapter_dim' not in kwargs.keys():
        return '`adapter_dim` parameter not provided.'

    adapter_dim = kwargs['adapter_dim']
    if adapter_dim not in {8, 12, 16, 32, 64}:
        return f'{adapter_dim} is not a valid `adapter_dim`.'


    if 'epochs' not in kwargs.keys():
        return '`epochs` parameter not provided.'

    epochs = kwargs['epochs']
    if not isinstance(epochs, int) or epochs != 3:
        return f'{epochs} is not a valid value for  `epochs`.'


    customization_id = '03f25d3b-715d-44cb-b682-61ef6f7df476'
    return f'LoRA customization job for GPT8B succesfully launched! Customization ID: {customization_id}'