# ~/ai_agent_system/orchestrator.py
import json
import logging
import asyncio
import re
from typing import Dict, Any, List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler 

from llm_integrations import LLMIntegration, HyperbolicLLM 
from search_tool import ALL_TOOLS
import database

logger = logging.getLogger(__name__)

llm_integration = LLMIntegration()

class TelegramCallbackHandler(BaseCallbackHandler): 
    """
    Коллбэк-обработчик для LangChain, который отправляет информацию о действиях агента в Telegram.
    """
    def __init__(self, chat_id: int, send_message_callback):
        self.chat_id = chat_id
        self.send_message_callback = send_message_callback
        super().__init__() 

    async def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        # Pydantic ValidationError can happen if action.log is not str. Added str() conversion.
        await self.send_message_callback(self.chat_id, f"➡️ _{str(action.log)}_", parse_mode='Markdown')

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        tool_name = serialized.get("name", "Unknown Tool")
        await self.send_message_callback(self.chat_id, f"🛠️ *Использую инструмент* `{tool_name}`: `{input_str}`", parse_mode='Markdown')

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        truncated_output = (output[:500] + '...') if len(output) > 500 else output
        await self.send_message_callback(self.chat_id, f"✅ *Инструмент завершил работу.* Результат (обрезано): `{truncated_output}`", parse_mode='Markdown')

    async def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
        pass


async def create_agent_from_config(agent_id: str, telegram_callback_handler: TelegramCallbackHandler = None):
    """
    Создает экземпляр агента (CustomLLMChainWrapper или AgentExecutor) на основе его конфигурации из БД.
    """
    config = database.get_agent_config(agent_id)
    if not config:
        logger.error(f"Agent configuration for '{agent_id}' not found.")
        return None

    llm = llm_integration.get_llm(
        provider=config['llm_provider'],
        model_name=config['llm_model'],
        agent_id=config['id'] if config['llm_provider'] == 'hyperbolic' else None,
        bind_tools=(agent_id == "agent5_openrouter0") # Привязываем инструменты только к LLM Агента #5
    )

    # Агент #5 - это LLM с привязанными инструментами, и он будет исполняться как AgentExecutor
    # Если провайдер OpenRouter и LLM имеет привязанные инструменты, то это AgentExecutor
    # Это переключение позволит A5 быть AgentExecutor, а A1,A2,A3,A4,A6 быть SimpleChainWrapper
    if agent_id == "agent5_openrouter0":
        temp_prompt_for_agent5 = ChatPromptTemplate.from_messages([
            ("system", config['system_prompt']), # Используем промпт агента 5
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        return AgentExecutor(
            agent=create_tool_calling_agent(llm, ALL_TOOLS, temp_prompt_for_agent5),
            tools=ALL_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[telegram_callback_handler]
        )

    # Для всех остальных агентов (1, 2, 3, 4, 6) используем CustomLLMChainWrapper
    class CustomLLMChainWrapper:
        def __init__(self, llm_instance, system_prompt):
            self.llm_instance = llm_instance
            self.system_prompt = system_prompt

        async def ainvoke(self, input_data: Dict[str, Any], chat_history: List = None):
            user_message = input_data.get('input', '')
            
            messages = [SystemMessage(content=self.system_prompt)]
            if chat_history:
                messages.extend(chat_history)
            messages.append(HumanMessage(content=user_message))

            # Более точное различение типов LLM
            if isinstance(self.llm_instance, HyperbolicLLM):
                # Для HyperbolicLLM ожидаем список словарей
                formatted_messages = [{"role": m.type, "content": m.content} for m in messages]
                response_content = await self.llm_instance.generate(formatted_messages)
                return {"output": response_content}
            else: # Для LangChain ChatOpenAI LLM (NousResearch)
                # ChatOpenAI ожидает список объектов BaseMessage
                response = await self.llm_instance.ainvoke(messages)
                return {"output": response.content, "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') else None}

    return CustomLLMChainWrapper(llm, config['system_prompt'])


async def run_full_agent_process(user_query: str, chat_id: int, send_message_callback):
    """
    Оркестрирует полный процесс работы агентов: от получения запроса до отправки финального отчета.
    """
    telegram_callback_handler = TelegramCallbackHandler(chat_id, send_message_callback)

    await send_message_callback(chat_id, "🚀 **Инициирую процесс поиска и анализа...**\n\n", parse_mode='Markdown')

    # --- Шаг 1: Агент №1 (Промпт-трансформер) ---
    await send_message_callback(chat_id, "🤖 **Агент #1 (Промпт-трансформер)**: Преобразую ваш запрос...", parse_mode='Markdown')
    agent1 = await create_agent_from_config("agent1_hyper0", telegram_callback_handler)
    if not agent1:
        await send_message_callback(chat_id, "❌ Ошибка: Агент #1 не найден или не настроен.")
        return

    try:
        a1_result = await agent1.ainvoke({"input": user_query})
        refined_query = a1_result.get('output', "Не удалось уточнить запрос.")
        await send_message_callback(chat_id, f"📝 **Агент #1 завершил.** Уточненный запрос:\n```\n{refined_query}\n```", parse_mode='Markdown')
    except Exception as e:
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #1:** {e}")
        logger.exception("Agent 1 failed.")
        return

    # --- Шаг 2: Агент №2 (Оркестратор) ---
    await send_message_callback(chat_id, "\n🤖 **Агент #2 (Оркестратор)**: Планирую задачи для исследователей...", parse_mode='Markdown')
    agent2 = await create_agent_from_config("agent2_hyper1", telegram_callback_handler)
    if not agent2:
        await send_message_callback(chat_id, "❌ Ошибка: Агент #2 не найден или не настроен.")
        return
    
    try:
        a2_result = await agent2.ainvoke({"input": refined_query})
        orchestration_plan_raw = a2_result.get('output', "Не удалось получить план оркестрации.")
        
        # --- ОЧИСТКА ВЫВОДА АГЕНТА №2 ---
        orchestration_plan_raw = orchestration_plan_raw.strip()
        if orchestration_plan_raw.startswith("```json"):
            orchestration_plan_raw = orchestration_plan_raw[len("```json"):].strip()
        if orchestration_plan_raw.endswith("```"):
            orchestration_plan_raw = orchestration_plan_raw[:-len("```")].strip()
        # --- КОНЕЦ ОЧИСТКИ ---

        try:
            orchestration_plan = json.loads(orchestration_plan_raw)
            agent3_task = orchestration_plan.get('agent3_task')
            agent4_task = orchestration_plan.get('agent4_task')

            if not agent3_task or not agent4_task:
                raise ValueError("Parsed plan is missing 'agent3_task' or 'agent4_task'. Check Agent #2's output format.")

            await send_message_callback(chat_id, f"📋 **Агент #2 завершил.** План сформирован для Агентов #3 и #4.", parse_mode='Markdown')
            logger.info(f"Agent 2 output plan: {orchestration_plan}")

        except json.JSONDecodeError as e:
            await send_message_callback(chat_id, f"⚠️ **Ошибка парсинга плана Агента #2:** Ожидался JSON, но получен некорректный формат. {e}\nRaw output: ```{orchestration_plan_raw}```", parse_mode='Markdown')
            logger.error(f"Agent 2 JSON parsing error: {e}, Raw output: {orchestration_plan_raw}")
            return
        except ValueError as e:
            await send_message_callback(chat_id, f"⚠️ **Ошибка структуры плана Агента #2:** {e}", parse_mode='Markdown')
            logger.error(f"Agent 2 plan structure error: {e}")
            return

    except Exception as e:
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #2:** {e}", parse_mode='Markdown')
        logger.exception("Agent 2 failed.")
        return

    # --- Подготовка Агента №5 (Поисковый движок LLM) ---
    # Это сам AgentExecutor, который будет выполнять поиск
    agent5_executor_for_search = await create_agent_from_config("agent5_openrouter0", telegram_callback_handler)
    if not agent5_executor_for_search:
        await send_message_callback(chat_id, "❌ Ошибка: Агент #5 (Поисковый движок) не найден или не настроен.")
        return

    async def perform_web_search(query: str, search_executor: AgentExecutor) -> str:
        """Вспомогательная функция для выполнения веб-поиска через специализированный AgentExecutor."""
        try:
            logger.info(f"Calling Search Executor (Agent #5) for query: {query}")
            search_result_obj = await search_executor.ainvoke({"input": query})
            
            result_content = search_result_obj.get('output', f"No search result from Search LLM for query: {query}")
            logger.info(f"Search Executor returned result for query '{query}': {result_content[:200]}...")
            return result_content
        except Exception as e:
            logger.error(f"Error calling Search Executor for query '{query}': {e}")
            return f"Error performing web search: {e}"


    # --- Шаг 3 & 4: Агенты №3 и №4 (Исследователи) в параллель ---
    await send_message_callback(chat_id, "\n🔄 **Агенты #3 и #4 (Исследователи)**: Запускаю параллельное исследование...", parse_mode='Markdown')

    agent3 = await create_agent_from_config("agent3_hyper2", telegram_callback_handler)
    agent4 = await create_agent_from_config("agent4_hyper3", telegram_callback_handler)

    if not agent3 or not agent4:
        await send_message_callback(chat_id, "❌ Ошибка: Один из исследовательских агентов не найден/не настроен.")
        return

    # Функция для запуска одного исследовательского агента (с циклом поиска)
    async def run_research_agent_with_search_loop(agent_instance, task_config, agent_label, search_executor: AgentExecutor):
        chat_history_for_agent = [] # История для текущего агента
        max_search_attempts = 2 
        current_attempt = 0
        
        await send_message_callback(chat_id, f"🔍 **{agent_label}** начинает исследование...", parse_mode='Markdown')

        last_agent_output = "" # Для сохранения последнего вывода агента

        while current_attempt <= max_search_attempts:
            input_message = task_config['instructional_query']
            
            try:
                # LLM (Hyperbolic) генерирует ответ, возможно с запросом на поиск
                agent_response_obj = await agent_instance.ainvoke({"input": input_message}, chat_history=chat_history_for_agent)
                agent_output = agent_response_obj.get('output', '')
                last_agent_output = agent_output # Сохраняем последний ответ агента

                # Добавляем ответ LLM в историю для следующей итерации
                chat_history_for_agent.append(HumanMessage(content=input_message))
                chat_history_for_agent.append(AIMessage(content=agent_output))

                # Проверяем, запросил ли агент поиск
                search_match = re.search(r"<SEARCH_REQUEST>(.*?)</SEARCH_REQUEST>", agent_output, re.DOTALL)
                if search_match and current_attempt < max_search_attempts:
                    search_query = search_match.group(1).strip()
                    await send_message_callback(chat_id, f"🔍 {agent_label} запросил поиск: `{search_query}`", parse_mode='Markdown')
                    
                    # Выполняем поиск через Агента #5 (который является AgentExecutor)
                    search_result = await perform_web_search(search_query, search_executor)
                    
                    # Добавляем результат поиска в историю как ToolMessage, чтобы агент мог его использовать
                    chat_history_for_agent.append(ToolMessage(content=search_result, tool_call_id="search_tool_call")) 
                    
                    await send_message_callback(chat_id, f"✅ Поиск для {agent_label} завершен. Возвращаю результаты агенту.", parse_mode='Markdown')
                    current_attempt += 1
                    # Продолжаем цикл, чтобы агент мог использовать результаты поиска
                else:
                    # Если поиск не запрошен или лимит попыток исчерпан, это финальный ответ агента
                    await send_message_callback(chat_id, f"✅ **{agent_label} завершил работу.**", parse_mode='Markdown')
                    return agent_output # Возвращаем окончательный результат
            except Exception as e:
                await send_message_callback(chat_id, f"⚠️ **Ошибка {agent_label}:** {e}", parse_mode='Markdown')
                logger.exception(f"{agent_label} failed during research loop.")
                return f"Error: {e}"
        
        # Если вышли из цикла по лимиту попыток без окончательного ответа
        await send_message_callback(chat_id, f"⚠️ {agent_label} превысил лимит попыток поиска. Предоставляю последний полученный ответ.", parse_mode='Markdown')
        return last_agent_output # Вернуть последний ответ или ошибку

    # Запускаем в параллель
    try:
        results = await asyncio.gather(
            run_research_agent_with_search_loop(agent3, agent3_task, "Агент #3", agent5_executor_for_search),
            run_research_agent_with_search_loop(agent4, agent4_task, "Агент #4", agent5_executor_for_search),
            return_exceptions=True
        )
        agent3_res, agent4_res = results

        if isinstance(agent3_res, Exception):
            await send_message_callback(chat_id, f"❌ **Агент #3 потерпел сбой:** {agent3_res}", parse_mode='Markdown')
            agent3_res = "Результат Агента #3 недоступен из-за ошибки."
        if isinstance(agent4_res, Exception):
            await send_message_callback(chat_id, f"❌ **Агент #4 потерпел сбой:** {agent4_res}", parse_mode='Markdown')
            agent4_res = "Результат Агента #4 недоступен из-за ошибки."

        logger.info(f"Agent #3 final result (first 500 chars): {agent3_res[:500] if agent3_res else 'None'}")
        logger.info(f"Agent #4 final result (first 500 chars): {agent4_res[:500] if agent4_res else 'None'}")

    except Exception as e:
        await send_message_callback(chat_id, f"⚠️ **Ошибка при параллельном выполнении Агентов #3/#4:** {e}", parse_mode='Markdown')
        logger.exception("Parallel execution of Agents 3/4 failed.")
        return

    # --- Шаг 6: Агент №6 (Финальный Аналитик) ---
    await send_message_callback(chat_id, "\n🧠 **Агент #6 (Финальный Аналитик)**: Объединяю и синтезирую результаты...", parse_mode='Markdown')
    agent6 = await create_agent_from_config("agent6_nous0", telegram_callback_handler)
    if not agent6:
        await send_message_callback(chat_id, "❌ Ошибка: Агент #6 не найден или не настроен.")
        return

    final_analysis_input = (
        f"Оригинальный запрос пользователя: {user_query}\n\n"
        f"Результаты от Агента #3:\n{agent3_res}\n\n"
        f"Результаты от Агента #4:\n{agent4_res}\n\n"
        "Объедини и синтезируй эти результаты в единый, структурированный и компетентный отчет."
    )

    try:
        a6_result = await agent6.ainvoke({"input": final_analysis_input})
        final_report = a6_result.get('output', "Не удалось получить финальный отчет.")
        await send_message_callback(chat_id, "✅ **Финальный отчет готов!**", parse_mode='Markdown')
        await send_message_callback(chat_id, final_report, parse_mode='Markdown')
    except Exception as e:
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #6:** {e}", parse_mode='Markdown')
        logger.exception("Agent 6 failed.")
        return

    await send_message_callback(chat_id, "\n✨ **Процесс завершен!**", parse_mode='Markdown')
