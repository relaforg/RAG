from openai import OpenAI


class QueryExpander:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="unused"
        )
        self.model = "qwen3:0.6b"

    def expand(self, prompt: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": (f"Generate 3 search query variants for: {prompt}"
                            "\nReturn one per line, no numbering.")
            }]
        )
        content = response.choices[0].message.content or ""
        return content.splitlines()
