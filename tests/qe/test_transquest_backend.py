from qe.transquest_backend import TransQuestBackend
from qe.service import QEService


def test_transquest_backend_returns_error_for_empty_input():
    backend = TransQuestBackend(model_name="some-model")

    result = backend.score("", "translated")

    assert result.score is None
    assert result.label is None
    assert result.error is not None


def test_transquest_backend_maps_score_to_high_confidence(monkeypatch):
    backend = TransQuestBackend(
        model_name="some-model",
        high_threshold=0.7,
        medium_threshold=0.4,
    )

    monkeypatch.setattr(backend, "_load_model", lambda: None)
    backend._model = object()

    monkeypatch.setattr(backend, "_predict_score", lambda s, t: 0.85)

    result = backend.score("verify your account", "verifizieren sie ihr konto")

    assert result.score == 0.85
    assert result.label == "high_confidence"
    assert result.error is None


def test_transquest_backend_maps_score_to_medium_confidence(monkeypatch):
    backend = TransQuestBackend(
        model_name="some-model",
        high_threshold=0.7,
        medium_threshold=0.4,
    )

    monkeypatch.setattr(backend, "_load_model", lambda: None)
    backend._model = object()

    monkeypatch.setattr(backend, "_predict_score", lambda s, t: 0.5)

    result = backend.score("verify your account", "verifizieren sie ihr konto")

    assert result.score == 0.5
    assert result.label == "medium_confidence"
    assert result.error is None


def test_transquest_backend_maps_score_to_low_confidence(monkeypatch):
    backend = TransQuestBackend(
        model_name="some-model",
        high_threshold=0.7,
        medium_threshold=0.4,
    )

    monkeypatch.setattr(backend, "_load_model", lambda: None)
    backend._model = object()

    monkeypatch.setattr(backend, "_predict_score", lambda s, t: 0.2)

    result = backend.score("verify your account", "verifizieren sie ihr konto")

    assert result.score == 0.2
    assert result.label == "low_confidence"
    assert result.error is None


def test_transquest_backend_returns_error_when_prediction_fails(monkeypatch):
    backend = TransQuestBackend(model_name="some-model")

    monkeypatch.setattr(backend, "_load_model", lambda: None)
    backend._model = object()

    def raise_error(source: str, target: str) -> float:
        raise RuntimeError("prediction exploded")

    monkeypatch.setattr(backend, "_predict_score", raise_error)

    result = backend.score("hello", "hallo")

    assert result.score is None
    assert result.label is None
    assert "prediction exploded" in result.error


def test_qe_service_rejects_disabled_transquest_backend():
    try:
        QEService.from_config(
            enable_qe=True,
            qe_backend="transquest",
            qe_model_name="some-model",
        )
    except ValueError as exc:
        assert "currently disabled" in str(exc)
        assert "future development" in str(exc)
    else:
        raise AssertionError("Expected transquest backend to be rejected")
