@app.post("/")
async def main(request: Request):
    body = await request.json()
    user_text = body["request"]["original_utterance"]

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=10
        )
        # Проверяем, успешен ли запрос
        if response.status_code != 200:
            error_detail = response.json() if response.text else "No detail"
            return {
                "response": {
                    "text": f"Ошибка DeepSeek API: {response.status_code} - {error_detail}"
                }
            }
        
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": answer
            }
        }
    except Exception as e:
        return {
            "response": {
                "text": f"Произошла ошибка: {str(e)}"
            }
        }
handler = app
