from typing import Protocol, List, Tuple, Any
from .data_classes import PromptWithLabel

class PromptWithExamplesCreator(Protocol):
    def __call__(self,
                 main_prompt: str,
                 system_context: str = "",
                 conversation_examples: List[PromptWithLabel] = []
                ) -> str:
        ...

class TextCleaner(Protocol):
    def __call__(self,
                 text: str,
                ) -> str:
        ...

class PromptWithLabelCreator(Protocol):
    def __call__(self,
                 data: Any,
                ) -> PromptWithLabel:
        ...