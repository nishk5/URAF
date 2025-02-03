import json
import requests
from loguru import logger
import re
import asyncio
import hashlib
import pickle
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True)

class LLMClient:
    """Client for interacting with LLM APIs."""

    def __init__(self, model, api_url, max_tokens=4000, temperature=0.5, top_p=0.85, top_k=50, min_p=0.2, presence_penalty=1.0):
        """Initialize LLM client with model settings."""
        self.model = model
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty

    def _generate_cache_key(self, prompt):
        """Generates a unique cache key based on the prompt."""
        data = json.dumps({"prompt": prompt}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _get_cache_path(self, cache_key):
        """Returns the file path for cached responses."""
        return CACHE_DIR / f"{cache_key}.pkl"

    def _load_from_cache(self, cache_key):
        """Loads response from cache if available."""
        cache_file = self._get_cache_path(cache_key)
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_key, response):
        """Saves response to cache."""
        with open(self._get_cache_path(cache_key), "wb") as f:
            pickle.dump(response, f)

    def clean_response(self, text):
        """Clean up LLM response."""
        if not text:
            return ""
            
        # Remove special tokens and artifacts
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'HeaderCode:.*?:', '', text, flags=re.DOTALL)
        text = re.sub(r'Response:.*?:', '', text, flags=re.DOTALL)
        text = re.sub(r'Cognitive Architecture:.*?:', '', text, flags=re.DOTALL)
        text = re.sub(r'and \*.*?\* (structure|sections)\.', '', text)
        
        # Remove leading/trailing whitespace and empty lines
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        # Ensure proper section headers
        if "*Understanding:*" not in text:
            text = "*Understanding:*\n" + text
            
        if "*Reasoning Pathway:*" not in text and "Reasoning:" in text:
            text = text.replace("Reasoning:", "*Reasoning Pathway:*")
            
        if "*Final Synthesis:*" not in text and "Synthesis:" in text:
            text = text.replace("Synthesis:", "*Final Synthesis:*")
            
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    async def query(self, prompt):
        """Query the LLM API."""
        cache_key = self._generate_cache_key(prompt)
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            return cached_response

        try:
            # Format request for LM Studio API
            data = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stream": False
            }
            
            # Disable proxy for localhost
            proxies = {
                "http": None,
                "https": None
            }
            
            response = requests.post(self.api_url, json=data, proxies=proxies)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"🔍 LLM Full API Response: {json.dumps(result, indent=2)}")
                
                if result and "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0].get("text", "").strip()
                    cleaned_text = self.clean_response(text)
                    
                    response_to_cache = {
                        "summary": cleaned_text,
                        "raw_text": prompt
                    }
                    self._save_to_cache(cache_key, response_to_cache)
                    
                    return response_to_cache
            
            logger.error(f"❌ LLM API Error: {response.status_code} - {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"❌ LLM Query Error: {str(e)}")
            return None

    async def batch_query(self, prompts):
        """
        Sends multiple queries in parallel.
        """
        tasks = [self.query(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        return responses
