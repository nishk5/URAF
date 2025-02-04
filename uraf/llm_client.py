import aiohttp
import asyncio
import json
import re
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import guidance
from .prompt_manager import PromptManager

class LLMClient:
    """
    Enforces structured LLM responses using guidance.
    """

    def __init__(self, model="qwen2.5-7b-instruct-1m", api_url="http://localhost:1234/v1/completions", 
                 max_tokens=4000, temperature=0.5, top_p=0.85, top_k=50):
        self.model = model
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

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
    async def query(self, prompt, technique=None):
        """Query the LLM API with guidance-based structured enforcement."""
        try:
            # Get structured prompt using guidance
            structured_prompt = PromptManager.get_structured_prompt(prompt, technique)
            
            # Format request for LM Studio API
            data = {
                "model": self.model,
                "prompt": str(structured_prompt),
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
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=data, proxies=proxies) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"🔍 LLM Full API Response: {json.dumps(result, indent=2)}")
                        
                        if result and "choices" in result and len(result["choices"]) > 0:
                            text = result["choices"][0].get("text", "").strip()
                            cleaned_text = self.clean_response(text)
                            
                            # Validate structure
                            if PromptManager.validate_structure(cleaned_text):
                                return {
                                    "summary": cleaned_text,
                                    "raw_text": prompt
                                }
                            else:
                                logger.warning("⚠️ LLM response did not follow the expected structure. Retrying...")
                                raise ValueError("Invalid response structure")
                    
                    logger.error(f"❌ LLM API Error: {response.status} - {await response.text()}")
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
