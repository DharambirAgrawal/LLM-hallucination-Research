__all__ = [
    "BaseModel",
    "OllamaModel",
    "OpenAICompatibleModel",
    "TransformersModel",
    "ReplayModel",
    "ModelFactory",
]


def __getattr__(name: str):
    """Load model adapters lazily so the stdlib API client needs no packages."""
    if name == "BaseModel":
        from .base_model import BaseModel
        return BaseModel
    if name == "OllamaModel":
        from .ollama_model import OllamaModel
        return OllamaModel
    if name == "OpenAICompatibleModel":
        from .openai_compatible_model import OpenAICompatibleModel
        return OpenAICompatibleModel
    if name == "TransformersModel":
        from .transformers_model import TransformersModel
        return TransformersModel
    if name == "ReplayModel":
        from .replay_model import ReplayModel
        return ReplayModel
    if name == "ModelFactory":
        from .model_factory import ModelFactory
        return ModelFactory
    raise AttributeError(name)
