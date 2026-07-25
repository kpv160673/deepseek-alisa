from fastapi import FastAPI, Request, HTTPException
import httpx
import os

app = FastAPI()

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

@app.post("/")
async def main(request: Request):
    # Чтение JSON с таймаутом (косвенно через лимит размера тела в FastAPI)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Некорректный JSON")

    # Валидация обязательных полей
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Тело запроса должно быть JSON‑объектом")

    req_data = body.get("request")
    if not isinstance(req_data, dict):
        raise HTTPException(status_code=400, detail="Отсутствует поле 'request'")

    user_text = req_data.get("original_utterance")
    if not user_text or not isinstance(user_text, str):
        raise HTTPException(status_code=400, detail="Поле 'original_utterance' должно быть непустой строкой")

    version = body.get("version")
    session = body.get("session")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                DEEPSEEK_API_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": user_text}],
                },
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Таймаут при запросе к DeepSeek API")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Ошибка соединения с DeepSeek API: {e}")

        # Обработка не‑200 ответов
        if response.status_code != 200:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text or "No detail"
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ошибка DeepSeek API: {response.status_code} - {error_detail}"
            )

        try:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise HTTPException(status_code=502, detail="Неожиданный формат ответа от DeepSeek API")

    return {
        "version": version,
        "session": session,
        "response": {
            "end_session": False,
            "text": answer
        }
    }
