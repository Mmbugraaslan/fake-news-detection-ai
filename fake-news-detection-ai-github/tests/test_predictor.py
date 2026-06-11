from app.services.predictor import PredictorService


def test_predictor_returns_prediction_shape() -> None:
    service = PredictorService()
    service.registry.register(
        "classical",
        type(
            "DummyModel",
            (),
            {
                "predict": staticmethod(
                    lambda text: {
                        "label": 0,
                        "label_name": "fake",
                        "score": 0.91,
                        "model_name": "classical",
                    }
                )
            },
        )(),
    )

    result = service.predict("Bu bir test haber metnidir.", model_name="classical")

    assert result["label"] == 0
    assert result["label_tr"] == "sahte"
    assert result["confidence"] == 0.91
    assert result["model"] == "classical"
    assert result["fallback_used"] is False