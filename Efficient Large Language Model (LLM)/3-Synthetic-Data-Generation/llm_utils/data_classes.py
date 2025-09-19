from typing import NamedTuple
from dataclasses import dataclass

class PromptWithLabel(NamedTuple):
    prompt: str
    label: str

@dataclass
class PtuningData:
    prompt: str
    completion: str

    def to_dict(self):
        return {"prompt": self.prompt, "completion": self.completion}
