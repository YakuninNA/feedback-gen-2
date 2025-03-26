import base64
import io
import json
from datetime import datetime

from arq.connections import create_pool
from config import redis_settings

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse

from starlette.templating import Jinja2Templates


router = APIRouter(
    prefix="/functionality",
    tags=["functionality"]
)


templates = Jinja2Templates(directory="/app/templates/gen_templates")


@router.get("/home", response_class=HTMLResponse)
async def functionality_home(
    request: Request
):
    return templates.TemplateResponse("functionality_page.html", {
        "request": request
    })


@router.get("/feedback_generation", response_class=HTMLResponse)
async def get_registration_form(
    request: Request,
):

    return templates.TemplateResponse("feedback_generation.html", {
        "request": request
    })


@router.post("/feedback_generation", response_class=HTMLResponse)
async def process_transcript(
        request: Request,
        transcript: UploadFile = File(...),
        requirements: str = Form(...),
        position: str = Form(...)
):
    try:
        # Read and process the file
        file_content = await transcript.read()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Connect to Redis
        redis = await create_pool(redis_settings)

        # Enqueue the job
        job = await redis.enqueue_job(
            'process_transcript_task',
            file_content.decode(),
            requirements,
            position,
            transcript.filename,
            timestamp
        )

        # Return the job_id to the frontend
        context = {
            "request": request,
            "message": "Your feedback is being generated. Please wait.",
            "job_id": job.job_id
        }

        return templates.TemplateResponse("feedback_popup.html", context)

    except Exception as e:
        return templates.TemplateResponse("feedback_generation.html", {
            "request": request,
            "error": f"An error occurred: {str(e)}."
        })


@router.get("/feedback_status/{job_id}", response_model=dict)
async def feedback_status(job_id: str):
    redis = await create_pool(redis_settings)
    result = await redis.get(job_id)

    if result:
        data = json.loads(result)
        if "error" in data:
            return {"status": "failed", "error": data["error"]}
        return {"status": "completed", "filename": data["filename"], "file_ready": True}
    return {"status": "pending", "file_ready": False}


@router.get("/download_feedback/{job_id}")
async def download_feedback(job_id: str):
    redis = await create_pool(redis_settings)
    result = await redis.get(job_id)

    if not result:
        raise HTTPException(status_code=404, detail="Job result not found or expired.")

    data = json.loads(result)
    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"])

    # Decode the base64-encoded file content
    file_content = base64.b64decode(data["file_content"])
    file_like = io.BytesIO(file_content)
    file_like.seek(0)

    return StreamingResponse(
        file_like,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{data["filename"]}"'}
    )


