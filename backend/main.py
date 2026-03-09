from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# load environment variables
load_dotenv()
print("API KEY:", os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# request structure
class CaptionRequest(BaseModel):
    niche: str
    topic: str
    tone: str


@app.get("/")
def home():
    return {"message": "Instagram AI Tool Running"}


@app.post("/generate_caption")
def generate_caption(data: CaptionRequest):

    prompt = f"""
    Generate an Instagram caption.

    Niche: {data.niche}
    Topic: {data.topic}
    Tone: {data.tone}

    Output format:

    Hook (1 sentence)

    Caption (3–4 short lines)

    Call To Action

    10 relevant hashtags
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"caption": response.choices[0].message.content}

@app.post("/generate_reel")
def generate_reel(data: CaptionRequest):

    prompt = f"""
You are generating a Reel script for Instagram.

Use EXACTLY the following inputs:

Niche: {data.niche}
Topic: {data.topic}
Tone: {data.tone}

Do NOT change the niche or topic.

Create a short reel idea.

Output format:

Hook (first 3 seconds)

Scene 1

Scene 2

Scene 3

Caption

10 hashtags related to the niche
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"reel_script": response.choices[0].message.content}


@app.post("/generate_calendar")
def generate_calendar(data: CaptionRequest):

    prompt = f"""
You are an Instagram content strategist.

Create a 7-day Instagram content calendar.

Inputs:
Niche: {data.niche}
Topic: {data.topic}
Tone: {data.tone}

Rules:
- Do NOT change the niche or topic
- Each day should include a post idea
- Mix content types (Reel, Carousel, Story, Post)

Output format:

Day 1:
Content Type:
Idea:

Day 2:
Content Type:
Idea:

Day 3:
Content Type:
Idea:

Day 4:
Content Type:
Idea:

Day 5:
Content Type:
Idea:

Day 6:
Content Type:
Idea:

Day 7:
Content Type:
Idea:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"calendar": response.choices[0].message.content}