from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
import os
import ollama
import warnings
from dataclasses import asdict


class OllamaTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("OLLAMA_API_KEY", None),
        provider: str = "ollama",
        organization: str = None,
        model: str = None,
        model_name: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", None)

        #if self.api_key is None:
        #    warnings.warn(
        #        "Ollama API key is not set. Please set the OPENAI_API_KEY environment variable."
        #    )

        self.model_name = model_name or "llama3.2:3b"
        self.model_max_token_dict = get_models_maxtoken_dict(models)
        
        for key,value in self.model_max_token_dict.items():
            print(f"{key : }{value}")
            

    def generate(
        self,
        messages: Union[List[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name

        ollama_config = {
            "model": self.model_name,
            "prompt": messages,
            "temperature": config.temperature,
            "k": config.top_k,
            "p": config.top_p,
            "num_generations": config.n,
            "stop_sequences": config.stop,
        }
        cache_key_params = ollama_config | {"messages": messages}
        
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                print("****** Using Cache ******")
                return TextGenerationResponse(**response)
        

        response = ollama.chat(model=model, messages=messages)
        response_gen = TextGenerationResponse(
                text=[dict(response.message)],
                config=ollama_config
        )
        cache_request(
            cache=self.cache, params=cache_key_params, values=asdict(response_gen)
        )
        return response_gen
        
    
    def count_tokens(self, text) -> int:
        numtk = num_tokens_from_messages(text)
        return num_tokens_from_messages(text)