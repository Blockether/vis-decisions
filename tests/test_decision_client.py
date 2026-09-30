"""The decision client uses the authenticated SDK transport, never a user gateway."""

import json

import pytest
from blockether.vis.engine import GatewayClient, GatewayError, ProtocolError, _client
from gateway_helpers import compatible, endpoint

from blockether.vis_decisions import Decisions


def test_decisions_infer_uses_existing_authenticated_gateway_client():
    def respond(method, path, body):
        if result := compatible(method, path, body):
            return result
        assert (method, path) == ("POST", "/v1/systemone")
        request = json.loads(body)
        assert set(request) == {"model", "state", "questions"}
        return 200, {
            "model": "laya-rl-agent",
            "routing": {"model": request["model"], "revision": "pinned"},
            "answers": {
                "truth": {
                    "type": "noul",
                    "noul": 0.625,
                    "confidence": 0.625,
                    "action": {"act_probability": 0.9},
                }
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }

    with endpoint(respond) as (url, calls):
        with GatewayClient(url, token="test-credential") as client:
            decisions = Decisions(client)
            questions = {"truth": {"type": "noul", "instructions": "Is this true?"}}
            for model in ("model-a", "model-b", "model-a"):
                result = decisions.infer(
                    model=model, state={"message": "yes"}, questions=questions
                )
                assert result["routing"]["model"] == model
                assert result["answers"]["truth"]["action"]["act_probability"] == 0.9
            with pytest.raises(ValueError, match="model is required"):
                decisions.infer(model="", state="yes", questions=questions)
        requests = [call for call in calls if call[1] == "/v1/systemone"]
        assert len(requests) == 3
        assert all(
            {k.lower(): v for k, v in call[2].items()}["authorization"]
            == "Bearer test-credential"
            for call in requests
        )


@pytest.mark.parametrize("model", ["gliner2.5-base", "gliner2.5-decide"])
def test_decision_input_limit_error_is_actionable_without_private_text(model):
    # #295: the SDK must retain safe counts instead of a generic HTTP 400.
    def respond(method, path, body):
        if result := compatible(method, path, body):
            return result
        assert (method, path) == ("POST", "/v1/systemone")
        return 400, {
            "error": {
                "type": "input-too-long",
                "message": "private state and test-credential must not appear",
                "input_tokens": 780,
                "max_input_tokens": 512,
                "state": "private state",
            }
        }

    with endpoint(respond) as (url, calls):
        with GatewayClient(url, token="test-credential") as gateway:
            with pytest.raises(GatewayError) as raised:
                Decisions(gateway).infer(
                    model=model,
                    state="Example task: " + "alpha beta gamma " * 250,
                    questions={
                        "priority": {
                            "type": "choice",
                            "instructions": "Select a label.",
                            "criteria": ["low", "high"],
                        }
                    },
                )
        error = raised.value
        assert error.status == 400
        assert error.code == "input-too-long"
        assert "780 tokens" in str(error)
        assert "limit is 512" in str(error)
        assert "Shorten" in str(error)
        assert error.input_tokens == 780
        assert error.max_input_tokens == 512
        assert "private state" not in repr(error)
        assert "test-credential" not in repr(error)
        assert sum(path == "/v1/systemone" for _, path, _, _ in calls) == 1


@pytest.mark.parametrize("field", ["input_tokens", "max_input_tokens"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, None, "test-credential", [], {}])
def test_decision_input_limit_rejects_invalid_diagnostics(field, value):
    # #295: do not expose arbitrary error bodies while adding actionable counts.
    payload = {
        "type": "input-too-long",
        "message": "private state and test-credential",
        "input_tokens": 780,
        "max_input_tokens": 512,
        field: value,
    }
    error = _client._gateway_error(
        400, json.dumps({"error": payload}).encode(), "test-credential"
    )
    assert error.code == "input-too-long"
    assert error.input_tokens is None
    assert error.max_input_tokens is None
    assert "Shorten" in str(error)
    assert "private state" not in repr(error)
    assert "test-credential" not in repr(error)


@pytest.mark.parametrize("code", ["input-too-long", "invalid-request"])
def test_input_limit_without_counts_preserves_safe_error_handling(code):
    error = _client._gateway_error(
        400,
        json.dumps({"error": {"type": code, "message": "test-credential"}}).encode(),
        "test-credential",
    )
    assert error.code == code
    assert error.input_tokens is None
    assert error.max_input_tokens is None
    assert "test-credential" not in repr(error)
    assert ("Shorten" in str(error)) == (code == "input-too-long")


@pytest.mark.parametrize("code", ["invalid-request", "input-too-long"])
def test_error_counts_do_not_expose_credentials_or_unrelated_diagnostics(code):
    error = _client._gateway_error(
        400,
        json.dumps(
            {
                "error": {
                    "type": code,
                    "message": "private state",
                    "input_tokens": 12345,
                    "max_input_tokens": 1234,
                }
            }
        ).encode(),
        "1234",
    )
    assert error.input_tokens is None
    assert error.max_input_tokens is None
    assert "1234" not in repr(error)
    assert "private state" not in repr(error)


def test_decision_model_catalog_uses_gateway_auth_and_preserves_separate_states():
    malformed = False

    def respond(method, path, body):
        if result := compatible(method, path, body):
            return result
        assert (method, path) == ("GET", "/v1/decisions/models")
        if malformed:
            return 200, {"models": {}}
        return 200, {
            "models": [
                {
                    "model_ref": "laya-typed-decisions",
                    "installed": True,
                    "residency": "cold",
                }
            ]
        }

    with endpoint(respond) as (url, calls):
        with GatewayClient(url, token="test-credential") as client:
            decisions = Decisions(client)
            assert decisions.list_models() == [
                {
                    "model_ref": "laya-typed-decisions",
                    "installed": True,
                    "residency": "cold",
                }
            ]
            malformed = True
            with pytest.raises(ProtocolError, match="models list"):
                decisions.list_models()
        requests = [call for call in calls if call[1] == "/v1/decisions/models"]
        assert len(requests) == 2
        assert all(
            {k.lower(): v for k, v in call[2].items()}["authorization"]
            == "Bearer test-credential"
            for call in requests
        )
