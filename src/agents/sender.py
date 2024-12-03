import yaml
import requests
from abc import ABC, abstractmethod

class BaseSender(ABC):
    def __init__(self, api_key):
        self.api_key = api_key

    @classmethod
    def load_api_keys(cls):
        with open('config.yaml') as config_file:
            config = yaml.safe_load(config_file)
        return config['api_keys']

    @abstractmethod
    def format_payload(self, message):
        pass

    @abstractmethod
    def send_request(self, message):
        pass

class ChatGPTSender(BaseSender):
    def format_payload(self, message):
        return {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": message}]
        }

    def send_request(self, message):
        payload = self.format_payload(message)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions",
                                  json=payload, headers=headers)
        return response.json()

class GeminiSender(BaseSender):
    def format_payload(self, message):
        return {
            "input": message,
            "key": self.api_key
        }

    def send_request(self, message):
        payload = self.format_payload(message)
        response = requests.post("https://api.gemini.com/v1/query", json=payload)
        return response.json()