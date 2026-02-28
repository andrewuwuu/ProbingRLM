from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .schemas import QueryRequest, QueryResponse
from .services import list_available_documents, run_query, stream_query_events

router = APIRouter(prefix="/api")


@router.get("/documents")
def get_documents():
    return {"documents": list_available_documents()}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    response, metrics = run_query(req)
    return QueryResponse(response=response, metrics=metrics)


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents selected.")

    return StreamingResponse(
        stream_query_events(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
