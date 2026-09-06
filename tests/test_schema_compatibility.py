"""The typing cleanup must not rename OpenAPI examples or request fields."""

import pytest
from pydantic import ValidationError

from schemas import OrchestratorRequest


def test_request_openapi_retains_legacy_example_metadata():
    properties = OrchestratorRequest.model_json_schema()["properties"]
    expected = {
        "type": "feedback",
        "conversation_id": "8db90ba1-aa03-494e-a46e-efddf7cb4277",
        "client_principal_id": "3d18e02b-d957-4cc5-85e6-e595cd53eec6",
        "client_principal_name": "jdoe@microsoft.com",
        "client_group_names": ["project-a", "admins"],
        "user_context": {},
    }
    for name, example in expected.items():
        assert properties[name]["example"] == example
        assert "examples" not in properties[name]


def test_feedback_and_question_request_contract_is_unchanged():
    question = OrchestratorRequest(question="question", user_context={"mode": "test"})
    assert question.model_dump()["question"] == "question"
    feedback = OrchestratorRequest(type="feedback", stars_rating=5)
    assert feedback.ask is None
    assert feedback.stars_rating == 5
    with pytest.raises(ValidationError):
        OrchestratorRequest(type="feedback", stars_rating=6)
