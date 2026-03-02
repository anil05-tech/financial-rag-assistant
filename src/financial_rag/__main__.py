"""Entry point: python -m financial_rag"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "financial_rag.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
