from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scraper import fetch_comments_from_url
from nlp_engine import analyze_post_comments

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str
    max_comments: int = 1000000  # Default to a high number to capture as many comments as possible

@app.post("/api/analyze-url")
def analyze_url(payload: URLRequest):
    platform, raw_comments = fetch_comments_from_url(payload.url)
    
    if not raw_comments:
        raise HTTPException(status_code=400, detail="No comments found or comments are disabled for this URL.")
        
    # Analyze actual fetched comments dynamically
    report = analyze_post_comments(payload.url, platform, raw_comments)
    return report