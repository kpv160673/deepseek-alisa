import os
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, Request, HTTPException
import httpx

# Настройка логирования (Vercel подхватит эти логи в панели Deployments → Logs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Чтение переменных окружения
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

logger.info("[INIT] Application started")
logger.info(f"[INIT] DEEPSEEK_API_KEY present: {bool(DEEPSEEK_API_KEY)}")
logger.info(f"[INIT] Using API URL: {DEEPSEEK_API_URL}")

def truncate_text(text: str, max_len: int = 1000) -> str:
    """Обрезает текст, чтобы не превысить лимиты Алисы."""
    if len(text) <= max_len:
        return text
    # Стараемся обрезать по слову, а не посередине
    return text[:max_len].rsplit(" ", 1)[0] + "…"

@app.post("/")
async def alice_skill(request: Request) -> Dict[str, Any]:
    # --- ОТЛАДОЧНЫЙ ЛОГ: функция вызвана ---
    logger.info("[HANDLER] Request received, starting processing")

    # Проверка наличия ключа уже внутри хендлера (чтобы функция запустилась даже без ключа)
    if not DEEPSEEK_API_KEY:
        logger.error("[HANDLER] ERROR: DEEPSEEK_API_KEY is missing")
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY не настроен на сервере")

    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"[HANDLER] JSON parse error: {e}")
        raise HTTPException(status_code=400, detail="Некорректный JSON в запросе")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Тело запроса должно быть JSON‑объектом")

    req = body.get("request")
    if not isinstance(req, dict):
        raise HTTPException(status_code=400, detail="Отсутствует поле 'request'")

    original_utterance = req.get("original_utterance")
    
    # Если пользователь просто активировал навык или сказал пустую фразу
    if not original_utterance or not isinstance(original_utterance, str) or not original_utterance.strip():
        logger.info("[HANDLER] Empty utterance, returning welcome message")
        return {
            "version": body.get("version", "1.0"),
            "session": body.get("session", {}),
            "response": {
                "text": "Привет! Я здесь. Задайте мне любой вопрос, и я расскажу всё, что знаю.",
                "end_session": False,
            },
        }

    session = body.get("session") or {}
    session_state = session.get("session_state") or {}
    
    # Получаем историю сообщений
    messages: List[Dict[str, str]] = session_state.get("messages", [])
    messages.append({"role": "user", "content": original_utterance})

    logger.info(f"[HANDLER] Sending request to DeepSeek with {len(messages)} messages in context")

    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            response = await client.post(
                DEEPSEEK_API_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                },
            )
        except httpx.TimeoutException:
            logger.error("[HANDLER] Timeout while calling DeepSeek API")
            raise HTTPException(status_code=504, detail="Таймаут при запросе к DeepSeek API")
        except Exception as e:
            logger.error(f"[HANDLER] Connection error to DeepSeek: {e}")
            raise HTTPException(status_code=502, detail=f"Ошибка соединения с DeepSeek API: {e}")

    # Обработка ошибок ответа API
    if response.status_code != 200:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text or "No detail"
        
        logger.error(f"[HANDLER] DeepSeek returned status {response.status_code}: {error_detail}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Ошибка DeepSeek API: {response.status_code} - {error_detail}",
        )

    try:
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"[HANDLER] Unexpected DeepSeek response format: {e}")
        raise HTTPException(status_code=502, detail="Неожиданный формат ответа от DeepSeek API")

    answer_text = truncate_text(answer)
    
    # Добавляем ответ модели в историю
    messages.append({"role": "assistant", "content": answer_text})

    # Ограничиваем длину истории, чтобы session_state не раздувался
    max_history_turns = 8
    if len(messages) > max_history_turns * 2:
        messages = messages[-max_history_turns * 2:]

    new_session_state = {"messages": messages}
    
    logger.info("[HANDLER] Response generated successfully")

    return {
        "version": body.get("version", "1.0"),
        "session": {
            **session,
            "session_state": new_session_state,
        },
        "response": {
            "text": answer_text,
            "end_session": False,
        },
    }
