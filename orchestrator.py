
#### 5. `orchestrator.py` (Значительные изменения в логике обработки Агентов 3, 4 и 5)

```python
# ~/ai_agent_system/orchestrator.py
import json
import logging
import asyncio
import re # Для парсинга SEARCH_REQUEST
from typing import Dict, Any, List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from llm_integrations import LLMIntegration
from search_tool import ALL_TOOLS # Список инструментов (только web_search)
import database

logger = logging.getLogger(__name__)

llm_integration = LLMIntegration()

class TelegramCallbackHandler:
    """
    Коллбэк-обработчик для LangChain, который отправляет информацию о действиях агента в Telegram.
    """
    def __init__(self, chat_id: int, send_message_callback):
        self.chat_id = chat_id
        self.send_message_callback = send_message_callback

    async def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        await self.send_message_callback(self.chat_id, f"➡️ _{action.log}_", parse_mode='Markdown')

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
    Создает экземпляр агента (LLM Chain или AgentExecutor) на основе его конфигурации из БД.
    """
    config = database.get_agent_config(agent_id)
    if not config:
        logger.error(f"Agent configuration for '{agent_id}' not found.")
        return None

    # Создаем LLM на основе конфигурации
    # Для Агента #5 (движок для поиска) мы привязываем инструменты (web_search) к его LLM.
    llm = llm_integration.get_llm(
        provider=config['llm_provider'],
        model_name=config['llm_model'],
        agent_id=config['id'] if config['llm_provider'] == 'hyperbolic' else None,
        bind_tools=(agent_id == "agent5_openrouter0") # Привязываем инструменты только к LLM Агента #5
    )

    # SimpleChainWrapper для всех LLM, которые не являются LangChain AgentExecutor
    # (Агенты 1, 2, 3, 4, 6, и Агент 5 - LLM с инструментом)
    class CustomLLMChainWrapper:
        def __init__(self, llm_instance, system_prompt):
            self.llm_instance = llm_instance
            self.system_prompt = system_prompt
            # Если llm_instance это ChatOpenAI с привязанными инструментами (Agent #5),
            # то его Chain для tool-calling будет создаваться при ainvoke.

        async def ainvoke(self, input_data: Dict[str, Any], chat_history: List = None):
            user_message = input_data.get('input', '')
            
            # Строим сообщения для LLM
            messages = [SystemMessage(content=self.system_prompt)]
            if chat_history:
                messages.extend(chat_history)
            messages.append(HumanMessage(content=user_message))

            # Если llm_instance это наш кастомный HyperbolicLLM
            if hasattr(self.llm_instance, 'generate'):
                # HyperbolicLLM ожидает формат dict for messages
                formatted_messages = [{"role": m.type, "content": m.content} for m in messages]
                response_content = await self.llm_instance.generate(formatted_messages)
                return {"output": response_content}
            # Если llm_instance это LangChain ChatOpenAI LLM (включая тот, что с bind_tools)
            else:
                response = await self.llm_instance.ainvoke(messages)
                # LangChain ChatOpenAI LLM с bind_tools может вернуть ToolCalls
                return {"output": response.content, "tool_calls": response.tool_calls}

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
    # Агент #5 - это LLM, который сам умеет использовать tool-calling (web_search)
    agent5_search_llm_chain = await create_agent_from_config("agent5_openrouter0", telegram_callback_handler)
    if not agent5_search_llm_chain:
        await send_message_callback(chat_id, "❌ Ошибка: Агент #5 (Поисковый движок) не найден или не настроен.")
        return

    async def perform_web_search(query: str) -> str:
        """Вспомогательная функция для выполнения веб-поиска через Агента №5."""
        try:
            logger.info(f"Calling Agent #5 (Search LLM) for query: {query}")
            # Агент #5 - это CustomLLMChainWrapper, который обертывает ChatOpenAI с bind_tools
            # LangChain автоматически обрабатывает ToolCalls, когда LLM их генерирует.
            # Если LLM вызывает инструмент, response.tool_calls будет заполнен,
            # и LangChain самостоятельно выполнит tool_call, а затем вернет его результат.
            # Мы просто передаем ему запрос, он сам решит, когда использовать web_search.
            
            # Для AgentExecutor LLM, мы вызываем .invoke или .ainvoke с историей сообщений.
            # Здесь, LLM (Agent #5) сам решит, когда вызвать web_search.
            
            # Для имитации LangChain AgentExecutor
            # Создадим временную цепь для LLM Агента 5 и привяжем к ней инструменты
            temp_agent5_llm_config = database.get_agent_config("agent5_openrouter0")
            temp_llm_with_tools = llm_integration.get_llm(
                provider=temp_agent5_llm_config['llm_provider'],
                model_name=temp_agent5_llm_config['llm_model'],
                bind_tools=True
            )
            # Создаем временный AgentExecutor для выполнения Tool Calling
            temp_prompt = ChatPromptTemplate.from_messages([
                ("system", temp_agent5_llm_config['system_prompt']),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            temp_agent_executor = AgentExecutor(
                agent=create_tool_calling_agent(temp_llm_with_tools, ALL_TOOLS, temp_prompt),
                tools=ALL_TOOLS,
                verbose=True,
                handle_parsing_errors=True,
                callbacks=[telegram_callback_handler] # Передаем коллбэки для видимости
            )

            # Выполняем запрос через этот временный AgentExecutor
            search_result_obj = await temp_agent_executor.ainvoke({"input": query})
            
            result_content = search_result_obj.get('output', f"No search result from Agent #5 for query: {query}")
            logger.info(f"Agent #5 (Search LLM) returned result for query '{query}': {result_content[:200]}...")
            return result_content
        except Exception as e:
            logger.error(f"Error calling Agent #5 (Search LLM) for query '{query}': {e}")
            return f"Error performing web search: {e}"


    # --- Шаг 3 & 4: Агенты №3 и №4 (Исследователи) в параллель ---
    await send_message_callback(chat_id, "\n🔄 **Агенты #3 и #4 (Исследователи)**: Запускаю параллельное исследование...", parse_mode='Markdown')

    agent3 = await create_agent_from_config("agent3_hyper2", telegram_callback_handler)
    agent4 = await create_agent_from_config("agent4_hyper3", telegram_callback_handler)

    if not agent3 or not agent4:
        await send_message_callback(chat_id, "❌ Ошибка: Один из исследовательских агентов не найден/не настроен.")
        return

    # Функция для запуска одного исследовательского агента (с циклом поиска)
    async def run_research_agent_with_search_loop(agent_instance, task_config, agent_label):
        chat_history = []
        max_search_attempts = 2 # Лимит попыток поиска для агента
        current_attempt = 0
        final_result = ""

        await send_message_callback(chat_id, f"🔍 **{agent_label}** начинает исследование...", parse_mode='Markdown')

        while current_attempt <= max_search_attempts:
            input_message = task_config['instructional_query']
            if current_attempt > 0 and final_result: # Если это не первая итерация и есть результат поиска
                input_message += f"\n\nИспользуйте следующие результаты поиска: {final_result}"
                chat_history.append(AIMessage(content=f"Результаты поиска: {final_result[:200]}..."))

            try:
                # LLM (Hyperbolic) генерирует ответ, возможно с запросом на поиск
                agent_response_obj = await agent_instance.ainvoke({"input": input_message}, chat_history=chat_history)
                agent_output = agent_response_obj.get('output', '')

                # Добавляем ответ LLM в историю
                chat_history.append(HumanMessage(content=input_message))
                chat_history.append(AIMessage(content=agent_output))

                # Проверяем, запросил ли агент поиск
                search_match = re.search(r"<SEARCH_REQUEST>(.*?)</SEARCH_REQUEST>", agent_output, re.DOTALL)
                if search_match and current_attempt < max_search_attempts:
                    search_query = search_match.group(1).strip()
                    await send_message_callback(chat_id, f"🔍 {agent_label} запросил поиск: `{search_query}`", parse_mode='Markdown')
                    
                    # Выполняем поиск через Агента #5
                    search_result = await perform_web_search(search_query)
                    final_result = search_result # Сохраняем результат поиска
                    
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
        return agent_output # Вернуть последний ответ или ошибку

    # Запускаем в параллель
    try:
        results = await asyncio.gather(
            run_research_agent_with_search_loop(agent3, agent3_task, "Агент #3"),
            run_research_agent_with_search_loop(agent4, agent4_task, "Агент #4"),
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
