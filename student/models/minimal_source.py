from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    text: str
    first_character_index: int
    last_character_index: int
