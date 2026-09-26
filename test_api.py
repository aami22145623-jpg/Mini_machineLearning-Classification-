from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"

def test_prediction():

    response = client.post(
        "/predict",
        json={
            "age": 35,
            "income_k": 75,
            "credit_score": 720
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "approved" in data
    assert "approval_probability" in data

if __name__ == "__main__":
    test_health()
    test_prediction()
    print("All tests passed!")