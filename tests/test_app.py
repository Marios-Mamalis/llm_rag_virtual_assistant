from fastapi.testclient import TestClient
from src.app import app


test_client = TestClient(app)


class TestCheckStatus:
    def test_check_status_code(self):
        assert test_client.get("/health").status_code == 200

    def test_check_status_message(self):
        assert test_client.get("/health").json() == {"status message": "Server is up."}
