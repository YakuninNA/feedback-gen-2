from fastapi import FastAPI

from genservice.routers import router as gen_router

app = FastAPI(
    title="FeedbackGeneratorApplication"
)

app.include_router(gen_router)

