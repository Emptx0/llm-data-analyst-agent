from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import uuid
import shutil

from src.agent import run_query

from src.config import DATA_DIR, BASE_DIR


router = APIRouter()

MAX_FILE_SIZE_MB = 50


@router.post("/generate")
async def generate(
    query: str = Form(...),
    file: UploadFile = File(...),
    verbose: bool = Form(False),
    max_steps: int = Form(8),
    max_new_tokens_plan: int = Form(256),
    max_new_tokens_tool: int = Form(256),
    max_new_tokens_final: int = Form(512),
):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    file_id = uuid.uuid4().hex
    temp_path = DATA_DIR / f"{file_id}.csv"


    # Save file
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file"
        ) from e


    # Size check
    try:
        size_mb = temp_path.stat().st_size / 1024**2
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to read uploaded file size"
        ) from e

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, "File too large")


    # Run agent
    try:
        result = run_query(
            user_query=query,
            dataset_path=str(temp_path),
            verbose=verbose,
            max_steps=max_steps,
            max_new_tokens_plan=max_new_tokens_plan,
            max_new_tokens_tool=max_new_tokens_tool,
            max_new_tokens_final=max_new_tokens_final,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response"
        ) from e


    return {
        "status": "ok",
        "result": result,
    }


