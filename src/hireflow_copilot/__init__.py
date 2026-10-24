"""HireFlow Copilot: RAG and a tool-using Claude agent over the HireFlow GraphQL API."""


def main() -> None:
    import uvicorn

    uvicorn.run("hireflow_copilot.main:app", host="0.0.0.0", port=8000)  # noqa: S104 - container entrypoint
