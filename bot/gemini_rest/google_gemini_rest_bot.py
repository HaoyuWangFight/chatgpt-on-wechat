# encoding:utf-8
import json

import requests

from bot.bot import Bot
from bot.session_manager import SessionManager
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf
from bot.baidu.baidu_wenxin_session import BaiduWenxinSession

BASE_URL = "https://gemini.relationshit.win"
API_VERSION = "v1beta"
MODEL = "gemini-1.5-pro-latest"


class GoogleGeminiRestBot(Bot):
    def __init__(self):
        super(GoogleGeminiRestBot, self).__init__()
        self.api_key = conf().get("gemini_api_key")
        # 复用文心的token计算方式
        self.sessions = SessionManager(BaiduWenxinSession, model=conf().get("model") or "gpt-3.5-turbo")

    def reply(self, query, context: Context = None) -> Reply:
        try:
            if context.type != ContextType.TEXT:
                logger.warn(f"[Gemini] Unsupported message type, type={context.type}")
                return Reply(ReplyType.TEXT, None)
            logger.info(f"[Gemini] query={query}")
            session_id = context["session_id"]
            session = self.sessions.session_query(query, session_id)
            gemini_messages = self._convert_to_gemini_messages(self.filter_messages(session.messages))
            url = f"{BASE_URL}/{API_VERSION}/models/{MODEL}:generateContent?key={self.api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "contents": gemini_messages,
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ],
                "generationConfig": {
                    "stopSequences": [
                        "Title"
                    ],
                    "temperature": 1.0,
                    "maxOutputTokens": 800,
                    "topP": 0.8,
                    "topK": 10
                }
            }
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                return Reply(ReplyType.TEXT, response.json()["candidates"][0]["content"]["parts"][0]["text"])
            else:
                logger.error(response.json())
                raise Exception(response.status_code)
        except Exception as e:
            logger.error("[Gemini] fetch reply error, may contain unsafe content")
            logger.error(e)
            return Reply(ReplyType.ERROR, "invoke [Gemini] api failed!")

    def _convert_to_gemini_messages(self, messages: list):
        res = []
        for msg in messages:
            if msg.get("role") == "user":
                role = "user"
            elif msg.get("role") == "assistant":
                role = "model"
            else:
                continue
            res.append({
                "role": role,
                "parts": [{"text": msg.get("content")}]
            })
        return res

    @staticmethod
    def filter_messages(messages: list):
        res = []
        turn = "user"
        if not messages:
            return res
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            if message.get("role") != turn:
                continue
            res.insert(0, message)
            if turn == "user":
                turn = "assistant"
            elif turn == "assistant":
                turn = "user"
        return res
