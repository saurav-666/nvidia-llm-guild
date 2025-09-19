import enum

class Models(enum.Enum):
    gpt8b = 'gpt-8b-000'
    gpt20b = 'gpt20b'
    gpt43b_2 = 'gpt-43b-002'
    gpt43b = 'gpt-43b-001'
    llama70b_chat = 'llama-2-70b-chat-hf'
    llama70b = 'llama-2-70b-hf'

    @classmethod
    def list_models(cls):
        for model in cls:
            print(f"{model.name}: {model.value}")

class PubmedModels(enum.Enum):
    gpt8b = 'gpt-8b-000'
    gpt20b = 'gpt20b'
    gpt43b = 'gpt-43b-001'

    @classmethod
    def list_models(cls):
        for model in cls:
            print(f"{model.name}: {model.value}")

class PtuneableModels(enum.Enum):
    gpt8b = 'gpt-8b-000'
    gpt20b = 'gpt20b'
    gpt43b = 'gpt-43b-002'
    llama70b = 'llama-2-70b-hf'

    @classmethod
    def list_models(cls):
        for model in cls:
            print(f"{model.name}: {model.value}")

class LoraModels(enum.Enum):
    gpt8b = 'gpt-8b-000-lora'
    gpt43b = 'gpt-43b-002-lora'

    @classmethod
    def list_models(cls):
        for model in cls:
            print(f"{model.name}: {model.value}")