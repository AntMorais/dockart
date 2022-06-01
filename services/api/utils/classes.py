from pydantic import BaseModel
from typing import List
from enum import Enum

class ModelType(str, Enum):
	style = "style"
	genre = "genre"
	artist = "artist"

class UserRequestIn(BaseModel):
	name: str
	path: str
	model_type: ModelType = "style"

class EntityOut(BaseModel):
    name: str
    description: str
    wiki_url: str

class EntitiesOut(BaseModel):
	entities: List[EntityOut]


class Properties(BaseModel):
    name: str = None
    model_type: str = None 