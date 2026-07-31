from typing import Dict, Optional
from pydantic import BaseModel, Field

class Profile(BaseModel):
    firstName: str = Field(default="John", alias="first_name")
    lastName: str = Field(default="Doe", alias="last_name")
    email: str = Field(default="john.doe@example.com")
    phone: str = Field(default="+1234567890")
    linkedin: Optional[str] = Field(default="https://linkedin.com/in/johndoe")
    github: Optional[str] = Field(default="https://github.com/johndoe")
    website: Optional[str] = Field(default="https://johndoe.dev")
    resumePath: Optional[str] = Field(default=None, alias="resume_path")
    customAnswers: Dict[str, str] = Field(default_factory=dict, alias="custom_answers")

    model_config = {
        "populate_by_name": True,
        "serialize_by_alias": False
    }
