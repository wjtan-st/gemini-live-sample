import asyncio
import base64
import json
import logging
import os

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from gemini_live import GeminiLive
from google.genai import types
from twilio_handler import TwilioHandler

# Load environment variables
load_dotenv()

# Configure logging - DEBUG for our modules, INFO for everything else
logging.basicConfig(level=logging.INFO)
logging.getLogger("gemini_live").setLevel(logging.DEBUG)
logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")
VOICE_NAME = os.getenv("VOICE_NAME", "Puck")

# Simple in-memory session resume data.
# This is process-local and single-user only — concurrent users will overwrite each other.
session_resume_data = {"handle": None, "intake_session_id": None}

# Twilio config (optional — only needed for phone call integration)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST")

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

GENIE_CHATBOT_URL = os.getenv("GENIE_CHATBOT_URL")
INTAKE_AGENT_UUID = os.getenv("INTAKE_AGENT_UUID")

# ─── Intake Agent Tool ────────────────────────────────────────────────────────
send_message_to_intake_agent_declaration = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="send_message_to_intake_agent",
            description=(
                "Sends the patient's response to the clinical intake agent for processing. "
                "Call this after every patient answer during the intake flow."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "message": types.Schema(
                        type=types.Type.STRING,
                        description="The patient's answer or message to forward to the intake agent.",
                    ),
                },
                required=["message"],
            ),
        )
    ]
)


def make_intake_agent_handler(intake_session: dict):
    """
    Returns a closure that calls the intake agent, injecting the session_id automatically.

    Args:
        intake_session (dict): The dictionary representing an intake_session which contains session_id.
    """

    async def send_message_to_intake_agent(message: str) -> dict:
        payload = {
            "audioIDs": [],
            "customFields": {},
            "documentIDs": [],
            "imageIDs": [],
            "userPrompt": message,
            "uuid": INTAKE_AGENT_UUID,
        }
        if intake_session["session_id"]:
            payload["sessionUUID"] = intake_session["session_id"]

        logger.info(
            f"Sending message to intake agent: session_id={intake_session['session_id']!r}, "
            f"message={message!r}"
        )
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                    GENIE_CHATBOT_URL,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response.raise_for_status()

                    answer = None
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if not data_str:
                            continue
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON SSE data: {data_str!r}")
                            continue

                        # Capture session_id (uuid) from the first event and persist globally
                        if not intake_session["session_id"] and event.get("uuid"):
                            intake_session["session_id"] = event["uuid"]
                            session_resume_data["intake_session_id"] = event["uuid"]
                            logger.info(f"Intake session_id set and persisted: {intake_session['session_id']}")

                        if event.get("status") == "completed":
                            answer = event.get("answer")
                            logger.info(f"Intake agent completed. answer={answer!r}")
                            break

                    if answer is None:
                        logger.warning("Intake agent SSE stream ended without a completed event.")
                        return {"error": "No completed response from intake agent."}

                    return {"answer": answer}

        except Exception as e:
            logger.error(f"Intake agent request failed: {e}")
            return {"error": str(e)}

    return send_message_to_intake_agent


async def send_message_to_intake_agent_mock(message: str, session_id: str) -> dict:
    """Mock intake agent: waits 15 seconds then returns a scripted guidance response."""
    logger.info(f"Intake agent mock called: session_id={session_id}, message={message!r}")
    await asyncio.sleep(15)
    result = {
        "guidance_text": [
            "Patient Mr. R has reported pain in his abdomen, try to gather more details, but ask for consent first.",
            "Thank the patient for agreeing to continue and for sharing when the pain started. Now ask where exactly in the abdomen they are feeling the pain — gently guide them to point to or describe the location.",
            "Acknowledge that the pain seems to be on the right side and that it is okay to be approximate. Then ask the patient to describe what the pain feels like — for example, sharp, dull, cramping, or burning.",
            "Acknowledge the cramping and sharp pain description, then ask whether the pain spreads or moves anywhere else — for example to the back, groin, or shoulder.",
            "Acknowledge that the pain stays in one spot and ask about any associated symptoms such as nausea, vomiting, fever, or changes in bowel habits.",
            "Acknowledge that nausea has been noted and ask whether the abdominal pain is constant, comes and goes, or arrives in waves.",
            "Acknowledge that the pain comes and goes rather than being constant, then ask the patient if anything makes the pain worse — such as movement, eating, or touching the area.",
            "Acknowledge that walking around seems to make the pain worse, then ask if anything helps relieve it — for example lying still, a particular position, or any medication taken.",
            "Acknowledge that staying still provides some relief. Ask the patient to rate their current pain on a scale of 0 to 10 to confirm the severity.",
            "Acknowledge that the patient has no known past medical conditions, then ask whether they have had any previous surgeries or operations.",
            "Acknowledge that the patient has no prior surgeries and ask if he is currently taking any medications, including prescriptions, over-the-counter drugs, or supplements.",
            "Acknowledge that the patient takes no medications, then ask if they have any known drug allergies or adverse reactions to any medications.",
            "Acknowledge that there is no relevant family history, then move on to ask about social history — including occupation, smoking and alcohol.",
            "Acknowledge the patient's social history (manufacturing work, non-smoker, occasional drinker) and ask whether they use any recreational drugs or substances to complete the social history section.",
            "Acknowledge the patient's answer without dwelling on the defensiveness — briefly reassure that all questions are standard and routine for every patient. Then smoothly transition to body systems review, asking about any cardiovascular or respiratory symptoms.",
            "Acknowledge that the patient has no other complaints, then ask briefly about respiratory symptoms such as cough, shortness of breath, or wheezing.",
            "Acknowledge the patient's response and move on to gastrointestinal symptoms. Note that nausea has already been mentioned; ask if there are any other GI symptoms such as vomiting, diarrhoea, constipation, or changes in bowel habits.",
            "Acknowledge the patient's response and move on to genitourinary symptoms. Ask about any pain or burning when urinating, changes in urinary frequency, or blood in the urine.",
            "Acknowledge the patient's answer and ask if there are any other symptoms they have noticed that haven't been mentioned yet, or proceed to close the body systems review and move toward wrapping up the interview.",
            "Thank the patient for answering all the questions. Let them know the information has been recorded and a doctor or nurse will be with them soon. Keep the message brief, warm, and reassuring."
        ]
    }
    logger.info(f"Intake agent mock response: {result}")
    return result


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, voice_name: str = VOICE_NAME, resume: bool = False):
    """WebSocket endpoint for Gemini Live."""
    await websocket.accept()

    session_resumption_handle = session_resume_data["handle"] if resume else None
    logger.info(
        f"WebSocket connection accepted with voice_name={voice_name}, resume={resume}, "
        f"has_resume_handle={bool(session_resumption_handle)}"
    )

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()

    async def audio_output_callback(data):
        await websocket.send_bytes(data)

    async def audio_interrupt_callback():
        # The event queue handles the JSON message, but we might want to do something else here
        pass

    def session_resumption_update_callback(handle):
        session_resume_data["handle"] = handle
        logger.info("Updated Live API session resumption handle")

    # Per-connection intake session state.
    # On resume, restore the intake session_id from global state so the
    # intake agent receives the correct sessionUUID instead of starting fresh.
    intake_session = {"session_id": session_resume_data["intake_session_id"] if resume else None}

    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY,
        model=MODEL,
        input_sample_rate=16000,
        voice_name=voice_name,
        session_resumption_handle=session_resumption_handle,
        on_session_resumption_update=session_resumption_update_callback,
        tools=[send_message_to_intake_agent_declaration],
        tool_mapping={"send_message_to_intake_agent": make_intake_agent_handler(intake_session)},
    )

    async def receive_from_client():
        try:
            while True:
                message = await websocket.receive()

                if message.get("bytes"):
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    text = message["text"]
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            logger.info(f"Received image chunk from client: {len(payload['data'])} base64 chars")
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                            continue
                    except json.JSONDecodeError:
                        pass

                    await text_input_queue.put(text)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")

    receive_task = asyncio.create_task(receive_from_client())

    async def run_session():
        async for event in gemini_client.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            text_input_queue=text_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
        ):
            if event:
                # Forward events (transcriptions, etc) to client
                await websocket.send_json(event)

    try:
        await run_session()
    except Exception as e:
        import traceback
        logger.error(f"Error in Gemini session: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        receive_task.cancel()
        # Ensure websocket is closed if not already
        try:
            await websocket.close()
        except:
            pass


# ─── Twilio Endpoints ─────────────────────────────────────────────────────────

@app.post("/twilio/inbound")
async def twilio_inbound():
    """Handles inbound Twilio calls. Returns TwiML to open a media stream."""
    host = TWILIO_APP_HOST or "localhost:8000"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Connecting to Gemini Live.</Say>
    <Connect>
        <Stream url="wss://{host}/twilio/stream" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.post("/twilio/outbound")
async def twilio_outbound(
    to_number: str = Query(..., description="Destination phone number (E.164 format)"),
    from_number: str = Query(..., description="Your Twilio phone number (E.164 format)"),
):
    """Initiates an outbound Twilio call that connects to Gemini Live."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return {"error": "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in environment"}
    if not TWILIO_APP_HOST:
        return {"error": "TWILIO_APP_HOST must be set in environment"}

    from twilio.rest import Client as TwilioClient

    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    twiml = f"""<Response>
    <Say>Connecting to Gemini Live.</Say>
    <Connect>
        <Stream url="wss://{TWILIO_APP_HOST}/twilio/stream" />
    </Connect>
</Response>"""

    call = client.calls.create(
        to=to_number,
        from_=from_number,
        twiml=twiml,
    )
    logger.info(f"Outbound call initiated: {call.sid}")
    return {"callSid": call.sid, "status": call.status}


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """WebSocket endpoint for Twilio Media Streams."""
    await websocket.accept()
    logger.info("Twilio media stream WebSocket connected")

    handler = TwilioHandler(gemini_api_key=GEMINI_API_KEY, model=MODEL)
    try:
        await handler.handle_media_stream(websocket)
    except Exception as e:
        logger.error(f"Twilio stream error: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Twilio media stream WebSocket closed")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
