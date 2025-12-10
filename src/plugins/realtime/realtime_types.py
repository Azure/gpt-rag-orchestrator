
from pydantic import BaseModel
from typing import Dict

class RTActionRequest(BaseModel):
    type: str
    payload: Dict