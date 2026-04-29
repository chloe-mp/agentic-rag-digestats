import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Le juge DeepEval — importé uniquement si deepeval est installé
try:
    from deepeval.models.base_model import DeepEvalBaseLLM

    gemini_pro = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    class GeminiJudge(DeepEvalBaseLLM):
        def __init__(self, model):
            self.model = model

        def load_model(self):
            return self.model

        def generate(self, prompt: str) -> str:
            return self.load_model().invoke(prompt).content

        async def a_generate(self, prompt: str) -> str:
            res = await self.load_model().ainvoke(prompt)
            return res.content

        def get_model_name(self):
            return "Gemini 2.5 Pro"

    judge = GeminiJudge(model=gemini_pro)
except ImportError:
    judge = None
