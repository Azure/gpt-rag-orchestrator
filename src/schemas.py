from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class OrchestratorRequest(BaseModel):
    # Core ask/question fields
    ask: Optional[str] = Field(
        None,
        description="Main user question or request. Optional here to allow pure feedback payloads; the API enforces it for non-feedback.",
    )
    question: Optional[str] = Field(
        None,
        description="Alias for 'ask', kept for backward compatibility. (Optional)"
    )

    # Operation type (e.g., feedback)
    type: Optional[str] = Field(
        None,
        description="Operation type. When set to 'feedback', the request is treated as a feedback submission.",
        example="feedback",
    )

    question_id: Optional[str] = Field(
        None,
        description="Identifier of the question within the conversation for which feedback is provided. (Optional)",
    )
    is_positive: Optional[bool] = Field(
        None,
        description="Thumbs up/down style flag for feedback. (Optional)",
    )
    stars_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Star rating from 1 to 5. (Optional)",
    )
    feedback_text: Optional[str] = Field(
        None,
        description="Free-form feedback text. (Optional)",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation identifier for keeping context between requests. (Optional)",
        example="8db90ba1-aa03-494e-a46e-efddf7cb4277"
    )
    client_principal_id: Optional[str] = Field(
        None,
        description="Unique ID of the authenticated user, if available. (Optional)",
        example="3d18e02b-d957-4cc5-85e6-e595cd53eec6"
    )
    client_principal_name: Optional[str] = Field(
        None,
        description="Display name of the authenticated user, if available. (Optional)",
        example="jdoe@microsoft.com"
    )
    client_group_names: Optional[List[str]] = Field(
        default_factory=list,
        description="List of groups the user belongs to, for authorization. (Optional)",
        example=['project-a', 'admins']
    )
    access_token: Optional[str] = Field(
        None,
        description="OAuth2 access token if authentication is provided. (Optional)"
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom user context to pass along to orchestrator. (Optional)",
        example={}
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Ask a question",
                    "value": {
                        "ask": "How often are performance reviews conducted at Contoso Electronics?",
                        "conversation_id": "8db90ba1-aa03-494e-a46e-efddf7cb4277",
                        "user_context": {}
                    },
                },
                {
                    "summary": "Submit feedback",
                    "value": {
                        "type": "feedback",
                        "conversation_id": "8db90ba1-aa03-494e-a46e-efddf7cb4277",
                        "question_id": "q-123",
                        "is_positive": True,
                        "stars_rating": 5,
                        "feedback_text": "Great answer, concise and accurate."
                    },
                },
            ]
        }


# Reusable OpenAPI responses for the orchestrator endpoint. Put here so
# route decorators in `main.py` remain compact and the examples are centralized.
ORCHESTRATOR_RESPONSES = {
    200: {
        "description": "OK — successful stream response (SSE). Each SSE 'data:' line contains a JSON object.",
        "content": {
            "text/event-stream": {
                "example": "gpt-rag answer"
            }
        }
    },
    400: {
        "description": "Bad Request — missing or invalid fields",
        "content": {
            "application/json": {
                "example": {"detail": "No 'ask' or 'question' field in request body"}
            }
        }
    },
    401: {
        "description": "Unauthorized — missing or invalid credentials",
        "content": {
            "application/json": {
                "example": {"detail": "Missing or invalid API key"}
            }
        }
    },
    403: {
        "description": "Forbidden — authenticated but not allowed to perform this action",
        "content": {
            "application/json": {
                "example": {"detail": "You are not authorized to perform this action"}
            }
        }
    },
    422: {
        "description": "Validation Error — request body did not match schema",
        "content": {
            "application/json": {
                "example": {"detail": [{"loc": ["body", "ask"], "msg": "field required", "type": "value_error.missing"}]}
            }
        }
    }
}
