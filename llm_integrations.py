# ~/ai_agent_system/llm_integrations.py
import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging
from search_tool import web_search # Импортируем наш инструмент

logger = logging.getLogger(__name__)

load_dotenv() # Загружаем переменные окружения

# Определение класса HyperbolicLLM вне метода get_llm, чтобы его можно было импортировать
class HyperbolicLLM:
    def __init__(self, model, api_key, temperature):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.url = "https://api.hyperbolic.xyz/v1/chat/completions"

    async def generate(self, messages):
        """
        Метод для генерации ответа Hyperbolic LLM.
        Ожидает список сообщений в формате [{"role": "user", "content": "..."}]
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "messages": messages,
            "model": self.model,
            "max_tokens": 1024,
            "temperature": self.temperature,
            "top_p": 0.9
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, headers=headers, json=data, timeout=120)
                response.raise_for_status()
                response_json = response.json()
                return response_json['choices'][0]['message']['content']
        except httpx.RequestError as e:
            logger.error(f"Hyperbolic API request failed: {e}")
            raise

class LLMIntegration:
    def __init__(self):
        self.hyperbolic_api_keys = {
            "agent1_hyper0": os.getenv("HYPERBOLIC_API_KEY_0"),
            "agent2_hyper1": os.getenv("HYPERBOLIC_API_KEY_1"),
            "agent3_hyper2": os.getenv("HYPERBOLIC_API_KEY_2"),
            "agent4_hyper3": os.getenv("HYPERBOLIC_API_KEY_3"),
        }
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.nousresearch_api_key = os.getenv("NOUSRESEARCH_API_KEY")
        
        logger.info(f"Loaded HYPERBOLIC_API_KEY_0: {'SET' if self.hyperbolic_api_keys.get('agent1_hyper0') else 'NOT SET'}")
        logger.info(f"Loaded OPENROUTER_API_KEY: {'SET' if self.openrouter_api_key else 'NOT SET'}")
        logger.info(f"Loaded NOUSRESEARCH_API_KEY: {'SET' if self.nousresearch_api_key else 'NOT SET'}")

    def get_llm(self, provider, model_name=None, agent_id=None, temperature=0.7, bind_tools=False):
        """
        Возвращает экземпляр LLM для указанного провайдера.
        """
        if provider == "hyperbolic":
            if not agent_id or agent_id not in self.hyperbolic_api_keys:
                raise ValueError(f"Hyperbolic agent_id '{agent_id}' not found in API keys or API key missing.")
            api_key = self.hyperbolic_api_keys[agent_id]
            if not api_key:
                raise ValueError(f"API key for Hyperbolic agent '{agent_id}' is not set (it's None/empty).")
            logger.info(f"Initializing Hyperbolic LLM with model: {model_name} for agent: {agent_id}")

            return HyperbolicLLM(model_name, api_key, temperature)

        elif provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key is not set.")
            logger.info(f"Initializing OpenRouter LLM with model: {model_name}, bind_tools={bind_tools}")
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature
            )
            if bind_tools:
                logger.info(f"Binding web_search tool to OpenRouter LLM: {model_name}")
                return llm.bind_tools([web_search])
            return llm

        elif provider == "nousresearch":
            if not self.nousresearch_api_key:
                raise ValueError("NousResearch API key is not set.")
            
            nous_base_url = "https://inference-api.nousresearch.com/v1" 
            logger.info(f"Initializing NousResearch LLM with model: {model_name} and base_url: {nous_base_url}")
            return ChatOpenAI(
                model=model_name,
                openai_api_key=self.nousresearch_api_key,
                base_url=nous_base_url, 
                temperature=temperature,
                request_timeout=180
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

# Пример использования (для тестирования)
if __name__ == '__main__':
    import asyncio
    llm_integration = LLMIntegration()
    async def test_llms():
        print("\n--- Testing LLM Initialization ---")
        try:
            # Test Hyperbolic
            hyper_llm_test = llm_integration.get_llm("hyperbolic", "meta-llama/Meta-Llama-3.1-405B-Instruct", "agent1_hyper0")
            print(f"Hyperbolic LLM initialized successfully. Type: {type(hyper_llm_test)}")
            if hasattr(hyper_llm_test, 'generate'):
                print("  (Has 'generate' method)")
                # result = await hyper_llm_test.generate([{"role": "user", "content": "Test Hyperbolic."}])
                # print(f"  Response: {result[:50]}...")

            # Test OpenRouter for search (agent 5's role)
            openrouter_search_llm_test = llm_integration.get_llm("openrouter", "x-ai/grok-3-mini", bind_tools=True)
            print(f"OpenRouter (Grok-3-mini) LLM with tool initialized successfully. Type: {type(openrouter_search_llm_test)}")
            if hasattr(openrouter_search_llm_test, 'bind_tools'):
                print("  (Has 'bind_tools' method)")
            if hasattr(openrouter_search_llm_test, 'ainvoke'):
                print("  (Has 'ainvoke' method)")

            # Test Nous Research
            nous_llm_test = llm_integration.get_llm("nousresearch", "Nous-Hermes-3.1-Llama-3.1-405B")
            print(f"NousResearch LLM initialized successfully. Type: {type(nous_llm_test)}")
            if hasattr(nous_llm_test, 'ainvoke'):
                print("  (Has 'ainvoke' method)")
            
        except ValueError as e:
            print(f"Error initializing LLMs: {e}")
        except Exception as e:
            print(f"General error during LLM test: {e}")
    
    asyncio.run(test_llms())
