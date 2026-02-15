from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import PyPDF2
import requests
import re
import os
import json
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import time
from typing import Optional, List
from supabase import create_client, Client
import hashlib
from datetime import datetime

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://ai-vision-guide.vercel.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
print("Loading YOLO model...")
yolo_model = YOLO("yolov8s.pt")
print("✓ YOLO model loaded")

# API configuration
HF_API_KEY = os.getenv("HF_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent"

# TTS Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
USE_ELEVENLABS = bool(ELEVENLABS_API_KEY)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase connected")
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}")
else:
    print("ℹ Supabase not configured (will use in-memory storage)")

# Global state (in-memory cache)
manual_context = {
    "manual_id": None,  # Current manual ID from Supabase
    "text": "",
    "full_manual_text": "",
    "parts_list": [],
    "steps": [],
    "current_step": 0,
    "images": []
}

detection_stats = {
    "total_requests": 0,
    "avg_response_time": 0,
    "last_detection_time": 0
}

# Supabase Helper Functions
def generate_manual_hash(text: str) -> str:
    """Generate unique hash for manual text"""
    return hashlib.sha256(text.encode()).hexdigest()

async def save_manual_to_supabase(filename: str, text: str, parts: List[dict], steps: List[dict]) -> Optional[str]:
    """Save manual data to Supabase, returns manual_id"""
    if not supabase:
        print("Supabase not configured, skipping save")
        return None
    
    try:
        # Generate unique hash to avoid duplicates
        text_hash = generate_manual_hash(text)
        
        # Check if manual already exists
        existing = supabase.table("manuals").select("id").eq("text_hash", text_hash).execute()
        
        if existing.data and len(existing.data) > 0:
            manual_id = existing.data[0]["id"]
            print(f"Manual already exists with ID: {manual_id}")
            return manual_id
        
        # Insert new manual
        manual_data = {
            "filename": filename,
            "text_hash": text_hash,
            "full_text": text,
            "parts": json.dumps(parts),
            "steps": json.dumps(steps),
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("manuals").insert(manual_data).execute()
        
        if result.data and len(result.data) > 0:
            manual_id = result.data[0]["id"]
            print(f"✓ Manual saved to Supabase with ID: {manual_id}")
            return manual_id
        
    except Exception as e:
        print(f"Error saving to Supabase: {e}")
        import traceback
        traceback.print_exc()
    
    return None

async def load_manual_from_supabase(manual_id: str) -> Optional[dict]:
    """Load manual data from Supabase by ID"""
    if not supabase:
        print("Supabase not configured")
        return None
    
    try:
        result = supabase.table("manuals").select("*").eq("id", manual_id).execute()
        
        if result.data and len(result.data) > 0:
            manual = result.data[0]
            print(f"✓ Manual loaded from Supabase: {manual.get('filename')}")
            
            return {
                "id": manual["id"],
                "filename": manual["filename"],
                "full_text": manual["full_text"],
                "parts": json.loads(manual["parts"]) if isinstance(manual["parts"], str) else manual["parts"],
                "steps": json.loads(manual["steps"]) if isinstance(manual["steps"], str) else manual["steps"]
            }
    except Exception as e:
        print(f"Error loading from Supabase: {e}")
    
    return None

async def list_manuals_from_supabase() -> List[dict]:
    """List all manuals from Supabase"""
    if not supabase:
        return []
    
    try:
        result = supabase.table("manuals").select("id, filename, created_at").order("created_at", desc=True).limit(50).execute()
        
        if result.data:
            return result.data
    except Exception as e:
        print(f"Error listing manuals: {e}")
    
    return []

class ImageData(BaseModel):
    image: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "en-US-Neural2-F"  # Default female voice

class AssistantQuery(BaseModel):
    question: str
    current_step: int
    detections: List[dict] = []
    frame: Optional[str] = None
    conversation_history: List[str] = []

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using Google Cloud TTS or ElevenLabs
    Falls back to edge-tts (free) if no API keys configured
    """
    
    # Try ElevenLabs first (highest quality)
    if USE_ELEVENLABS:
        try:
            response = requests.post(
                "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "text": request.text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return StreamingResponse(
                    BytesIO(response.content),
                    media_type="audio/mpeg"
                )
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
    
    # Fallback to Google Cloud TTS via public API
    try:
        import gtts
        tts = gtts.gTTS(text=request.text, lang='en', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg"
        )
    except ImportError:
        print("gTTS not installed, trying edge-tts...")
    except Exception as e:
        print(f"gTTS error: {e}")
    
    # Fallback to edge-tts (Microsoft Edge TTS - free and good quality)
    try:
        import edge_tts
        import asyncio
        
        audio_buffer = BytesIO()
        
        async def generate_audio():
            communicate = edge_tts.Communicate(
                request.text, 
                "en-US-AriaNeural"  # Natural female voice
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
        
        asyncio.run(generate_audio())
        audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg"
        )
    except ImportError:
        print("edge-tts not installed. Install with: pip install edge-tts")
    except Exception as e:
        print(f"edge-tts error: {e}")
    
    # If all else fails, return error
    return {"error": "TTS not configured. Install: pip install gtts or pip install edge-tts"}


@app.post("/detect")
def detect(data: ImageData):
    """Optimized YOLO detection"""
    start_time = time.time()
    
    try:
        img_data = data.image.split(",")[1] if "," in data.image else data.image
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        results = yolo_model(
            frame,
            conf=0.25,
            iou=0.45,
            max_det=10,
            verbose=False
        )
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = yolo_model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "color": "#22c55e"
                })
        
        elapsed = time.time() - start_time
        detection_stats["total_requests"] += 1
        detection_stats["avg_response_time"] = (
            (detection_stats["avg_response_time"] * (detection_stats["total_requests"] - 1) + elapsed) 
            / detection_stats["total_requests"]
        )
        detection_stats["last_detection_time"] = elapsed
        
        return {
            "detections": detections,
            "processing_time_ms": int(elapsed * 1000),
            "frame_size": f"{width}x{height}"
        }
        
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return {"detections": [], "error": str(e)}

@app.post("/detect-with-manual")
def detect_with_manual(data: ImageData):
    """Manual-aware detection"""
    start_time = time.time()
    
    try:
        current_step = manual_context.get("current_step", 0)
        steps = manual_context.get("steps", [])
        parts = manual_context.get("parts_list", [])
        
        current_instruction = ""
        if steps and current_step < len(steps):
            current_instruction = steps[current_step]["instruction"]
        
        yolo_result = detect(data)
        detections = yolo_result.get("detections", [])
        
        if parts and detections:
            part_keywords = set()
            for part in parts:
                words = part['name'].lower().split()
                part_keywords.update(words)
            
            for det in detections:
                label_lower = det['label'].lower()
                if any(keyword in label_lower for keyword in part_keywords):
                    det['color'] = "#a855f7"
                    det['highlight'] = True
                    det['confidence'] = min(det['confidence'] + 0.2, 1.0)
        
        instruction = ""
        if current_instruction and detections:
            detected_labels = [d['label'] for d in detections[:3]]
            instruction = f"Step {current_step + 1}: {current_instruction[:100]}"
        elif detections:
            labels = [d['label'] for d in detections[:3]]
            instruction = f"Found {len(detections)} items: {', '.join(labels)}"
        else:
            instruction = "Point camera at the parts for the current step"
        
        elapsed = time.time() - start_time
        
        return {
            "detections": detections,
            "instruction": instruction,
            "current_step": current_step,
            "total_steps": len(steps),
            "processing_time_ms": int(elapsed * 1000),
            "current_step_detail": steps[current_step] if steps and current_step < len(steps) else None
        }
        
    except Exception as e:
        print(f"Manual detection error: {str(e)}")
        return detect(data)

@app.post("/ask-assistant")
async def ask_assistant(query: AssistantQuery):
    """LLM-powered voice assistant with live object detection context"""
    
    if not GEMINI_API_KEY:
        return {
            "answer": "Sorry, AI assistant is not configured. Please set GEMINI_API_KEY.",
            "highlight_objects": [],
            "step_instruction": None
        }
    
    try:
        # Get manual context
        manual_text = manual_context.get("full_manual_text", "")
        parts_list = manual_context.get("parts_list", [])
        steps = manual_context.get("steps", [])
        current_step = query.current_step
        
        # Build context
        current_step_info = ""
        if steps and current_step < len(steps):
            current_step_info = f"Current Step {current_step + 1}: {steps[current_step]['instruction']}"
        
        parts_context = ""
        if parts_list:
            parts_context = "Parts List:\n"
            for part in parts_list[:15]:
                parts_context += f"- {part['name']} (ID: {part['id']}, Qty: {part['quantity']})\n"
        
        steps_context = ""
        if steps:
            steps_context = "Assembly Steps:\n"
            for i, step in enumerate(steps[:10], 1):
                prefix = "→ " if i == current_step + 1 else "  "
                steps_context += f"{prefix}Step {step['number']}: {step['instruction']}\n"
        
        # Live detection context
        detection_context = ""
        if query.detections:
            detected_objects = [d['label'] for d in query.detections]
            detection_context = f"\nCurrently visible in camera: {', '.join(detected_objects)}"
        
        # Create prompt
        prompt = f"""You are an expert AI assembly assistant with LIVE camera vision. You can see what the user is looking at in real-time through object detection.

MANUAL CONTEXT:
{parts_context}

{steps_context}

{current_step_info}

{detection_context}

FULL MANUAL (for detailed reference):
{manual_text[:6000]}

USER QUESTION/COMMAND: {query.question}

YOUR TASK:
1. Answer questions clearly based on the manual and what you see in the camera
2. Reference specific parts by ID when relevant
3. If they ask "where is X", check if X is in the detected objects and guide them
4. If they're stuck, break down the step into smaller actions
5. Be encouraging and supportive - assembly can be challenging!
6. If they say "start tutorial", explain they should use the app's tutorial mode
7. Keep answers conversational and under 100 words

RESPONSE FORMAT (JSON):
{{
  "answer": "Your helpful spoken response (2-3 sentences, natural language)",
  "highlight_objects": ["list", "of", "object", "labels", "from", "camera", "to", "highlight"],
  "step_instruction": "Optional: brief overlay text (1 sentence, only if giving specific action)"
}}

IMPORTANT:
- Keep answer BRIEF for voice output (under 100 words)
- Use simple, clear language - you're speaking to them
- Only include objects in highlight_objects that are ACTUALLY in the detected objects list
- step_instruction should be SHORT (under 10 words) as it's displayed as overlay
- Be conversational - you're their helpful guide, not a robot

Respond with ONLY the JSON, no other text."""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 512
            }
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                generated_text = generated_text.replace('```json', '').replace('```', '').strip()
                
                try:
                    parsed_response = json.loads(generated_text)
                    
                    return {
                        "answer": parsed_response.get("answer", "I couldn't find that information."),
                        "highlight_objects": parsed_response.get("highlight_objects", []),
                        "step_instruction": parsed_response.get("step_instruction", None)
                    }
                except json.JSONDecodeError:
                    return {
                        "answer": generated_text[:300],
                        "highlight_objects": [],
                        "step_instruction": None
                    }
        else:
            print(f"Gemini API Error: {response.status_code}")
            return {
                "answer": "Sorry, I'm having trouble connecting. Please try again.",
                "highlight_objects": [],
                "step_instruction": None
            }
            
    except Exception as e:
        print(f"Assistant error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "answer": "I encountered an error. Please try rephrasing your question.",
            "highlight_objects": [],
            "step_instruction": None
        }

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        print(f"Extracted {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def extract_images_from_pdf(pdf_bytes):
    """Extract images from PDF"""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150, fmt='jpeg')
        image_list = []
        for i, img in enumerate(images):
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_list.append({
                'page': i + 1,
                'data': f"data:image/jpeg;base64,{img_base64}"
            })
        print(f"Extracted {len(image_list)} page images")
        return image_list
    except Exception as e:
        print(f"Error extracting images: {str(e)}")
        return []

def parse_manual_with_gemini(text):
    """Parse manual with Gemini AI"""
    if not GEMINI_API_KEY:
        return parse_manual_basic(text)
    
    try:
        prompt = f"""Parse this assembly manual and extract parts and steps.

Manual Text:
{text[:8000]}

Return ONLY valid JSON:
{{
  "parts": [{{"id": "A", "name": "Long screw", "quantity": 4}}],
  "steps": [{{"number": 1, "instruction": "Attach panel A to base"}}]
}}"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                generated_text = generated_text.replace('```json', '').replace('```', '').strip()
                parsed_data = json.loads(generated_text)
                
                parts_list = parsed_data.get('parts', [])
                steps = parsed_data.get('steps', [])
                
                print(f"Gemini parsed: {len(parts_list)} parts, {len(steps)} steps")
                return parts_list, steps
            
    except Exception as e:
        print(f"Gemini parsing failed: {str(e)}")
    
    return parse_manual_basic(text)

def parse_manual_basic(text):
    """Basic fallback parsing"""
    parts_list = []
    steps = []
    
    if not text or len(text.strip()) < 50:
        return parts_list, steps
    
    # Extract parts
    patterns = [
        r'([A-Z]\d*|\d+[A-Z]?)\s*[-:\.]\s*([^\n]+?)(?:\s*\((\d+)\s*(?:pcs?|pieces?)?\))?',
        r'[•\-\*]\s*([^\n:]+?):\s*([^\n]+?)(?:\s*\((\d+)\s*(?:pcs?|pieces?)?\))?',
        r'(\d+)\s*x\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if len(match) >= 2:
                part_id = match[0].strip() if match[0] else f"P{len(parts_list)+1}"
                part_name = match[1].strip()
                quantity = match[2].strip() if len(match) > 2 and match[2] else "1"
                
                if len(part_name) > 3:
                    parts_list.append({
                        "id": part_id,
                        "name": part_name[:80],
                        "quantity": quantity
                    })
    
    # Extract steps
    step_patterns = [
        r'(?:STEP\s*)?(\d+)\s*[:\.\-]\s*([^\n]+(?:\n(?!\s*(?:STEP\s*)?\d+\s*[:\.\-])[^\n]+)*)',
        r'(\d+)\)\s*([^\n]+)',
        r'Step\s+(\d+):\s*([^\n]+)',
    ]
    
    for pattern in step_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            step_num = int(match[0])
            instruction = match[1].strip()
            
            if len(instruction) > 10:
                steps.append({
                    "number": step_num,
                    "instruction": instruction[:250]
                })
    
    parts_list = list({p['name']: p for p in parts_list}.values())
    steps = sorted(steps, key=lambda x: x['number'])
    
    print(f"Basic parsing: {len(parts_list)} parts, {len(steps)} steps")
    return parts_list, steps

@app.post("/upload-manual")
async def upload_manual(file: UploadFile = File(...)):
    """Upload and process manual, save to Supabase"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {file.filename}")
        print(f"{'='*60}")
        
        pdf_content = await file.read()
        pdf_file = BytesIO(pdf_content)
        
        print("Extracting text...")
        text = extract_text_from_pdf(pdf_file)
        
        print("Extracting images...")
        pdf_file.seek(0)
        page_images = extract_images_from_pdf(pdf_content)
        
        print("Parsing with AI...")
        parts_list, steps = parse_manual_with_gemini(text)
        
        # Save to Supabase
        manual_id = await save_manual_to_supabase(file.filename, text, parts_list, steps)
        
        # Store in memory cache
        manual_context["manual_id"] = manual_id
        manual_context["text"] = text[:1000]
        manual_context["full_manual_text"] = text
        manual_context["parts_list"] = parts_list
        manual_context["steps"] = steps
        manual_context["current_step"] = 0
        manual_context["images"] = page_images
        
        print(f"\n{'='*60}")
        print(f"✓ Success!")
        print(f"  Manual ID: {manual_id}")
        print(f"  Parts: {len(parts_list)}")
        print(f"  Steps: {len(steps)}")
        print(f"  Text: {len(text)} chars")
        print(f"  Storage: {'Supabase' if manual_id else 'Memory only'}")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "manual_id": manual_id,
            "content": f"{len(parts_list)} parts, {len(steps)} steps",
            "parts_count": len(parts_list),
            "steps_count": len(steps),
            "parts": parts_list[:10],
            "steps": steps,
            "first_step": steps[0] if steps else None
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "parts_count": 0,
            "steps_count": 0
        }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "yolo_loaded": yolo_model is not None,
        "llm_ready": bool(GEMINI_API_KEY),
        "manual_loaded": bool(manual_context.get("full_manual_text")),
        "supabase_connected": supabase is not None
    }

@app.get("/stats")
def get_stats():
    return {
        "total_detections": detection_stats["total_requests"],
        "avg_response_time_ms": int(detection_stats["avg_response_time"] * 1000),
        "manual_loaded": bool(manual_context.get("full_manual_text")),
        "steps_count": len(manual_context.get("steps", [])),
        "manual_id": manual_context.get("manual_id")
    }

@app.get("/manuals")
async def get_manuals():
    """List all uploaded manuals from Supabase"""
    if not supabase:
        return {
            "success": False,
            "error": "Supabase not configured",
            "manuals": []
        }
    
    manuals = await list_manuals_from_supabase()
    return {
        "success": True,
        "manuals": manuals
    }

@app.post("/load-manual/{manual_id}")
async def load_manual(manual_id: str):
    """Load a previously uploaded manual from Supabase"""
    if not supabase:
        return {
            "success": False,
            "error": "Supabase not configured"
        }
    
    manual_data = await load_manual_from_supabase(manual_id)
    
    if not manual_data:
        return {
            "success": False,
            "error": "Manual not found"
        }
    
    # Load into memory cache
    manual_context["manual_id"] = manual_data["id"]
    manual_context["text"] = manual_data["full_text"][:1000]
    manual_context["full_manual_text"] = manual_data["full_text"]
    manual_context["parts_list"] = manual_data["parts"]
    manual_context["steps"] = manual_data["steps"]
    manual_context["current_step"] = 0
    
    return {
        "success": True,
        "manual_id": manual_data["id"],
        "filename": manual_data["filename"],
        "parts_count": len(manual_data["parts"]),
        "steps_count": len(manual_data["steps"]),
        "parts": manual_data["parts"][:10],
        "steps": manual_data["steps"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "="*60)
    print("🤖 AI Vision Guide - Voice Assistant Edition")
    print("="*60)
    print("Features:")
    print("  ✓ Real-time YOLO detection")
    print(f"  {'✓' if GEMINI_API_KEY else '✗'} Voice Q&A with LLM")
    print(f"  {'✓' if GEMINI_API_KEY else '✗'} Tutorial mode")
    print("\nStorage:")
    if supabase:
        print("  ✓ Supabase connected - manuals persist across sessions")
    else:
        print("  ℹ Using memory storage - manuals reset on restart")
        print("  ℹ To enable: Set SUPABASE_URL and SUPABASE_KEY")
    print("\nTTS Configuration:")
    if USE_ELEVENLABS:
        print("  ✓ ElevenLabs TTS (Premium)")
    else:
        print("  ℹ Using free TTS (gtts/edge-tts)")
        print("  ℹ For better voice: export ELEVENLABS_API_KEY='your_key'")
    print("\nEndpoints:")
    print("  • POST /upload-manual - Upload PDF (saves to Supabase)")
    print("  • GET /manuals - List all manuals")
    print("  • POST /load-manual/{id} - Load manual by ID")
    print("  • POST /detect-with-manual - Live detection")
    print("  • POST /ask-assistant - Voice Q&A")
    print("  • POST /tts - Text-to-speech")
    print("\nRequired:")
    print("  pip install supabase  (for persistent storage)")
    print("  pip install gtts  OR  pip install edge-tts  (for TTS)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)