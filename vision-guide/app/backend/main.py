from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
from functools import lru_cache

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://ai-vision-guide.vercel.app",
        "*"  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once at startup
print("Loading YOLO model...")
yolo_model = YOLO("yolov8s.pt")
print("✓ YOLO model loaded")

# API configuration
HF_API_KEY = os.getenv("HF_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Global state
manual_context = {
    "text": "",
    "parts_list": [],
    "steps": [],
    "current_step": 0,
    "images": []
}

# Performance tracking
detection_stats = {
    "total_requests": 0,
    "avg_response_time": 0,
    "last_detection_time": 0
}

class ImageData(BaseModel):
    image: str

# Optimized YOLO detection with caching
@app.post("/detect")
def detect(data: ImageData):
    """Optimized YOLO detection for real-time video"""
    start_time = time.time()
    
    try:
        # Decode base64 image
        img_data = data.image.split(",")[1] if "," in data.image else data.image
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Resize for faster processing (optional - adjust based on accuracy needs)
        # Smaller size = faster, but less accurate
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Run YOLO detection with optimized settings
        results = yolo_model(
            frame,
            conf=0.25,  # Lower confidence threshold
            iou=0.45,   # NMS IOU threshold
            max_det=10,  # Limit max detections for speed
            verbose=False  # Disable verbose output
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
        
        # Update stats
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
    """Manual-aware detection - optimized for real-time"""
    start_time = time.time()
    
    try:
        # Get current step info
        current_step = manual_context.get("current_step", 0)
        steps = manual_context.get("steps", [])
        parts = manual_context.get("parts_list", [])
        
        current_instruction = ""
        if steps and current_step < len(steps):
            current_instruction = steps[current_step]["instruction"]
        
        # Run base YOLO detection (fast)
        yolo_result = detect(data)
        detections = yolo_result.get("detections", [])
        
        # Filter/enhance detections based on manual context
        if parts and detections:
            # Create a set of part keywords from manual
            part_keywords = set()
            for part in parts:
                words = part['name'].lower().split()
                part_keywords.update(words)
            
            # Boost confidence for detected objects that match manual parts
            for det in detections:
                label_lower = det['label'].lower()
                if any(keyword in label_lower for keyword in part_keywords):
                    det['color'] = "#a855f7"  # Purple for manual-relevant items
                    det['highlight'] = True
                    det['confidence'] = min(det['confidence'] + 0.2, 1.0)  # Boost confidence
        
        # Generate instruction
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
        return detect(data)  # Fallback to regular detection

# PDF Processing functions (unchanged but optimized)
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF manual"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        print(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def extract_images_from_pdf(pdf_bytes):
    """Extract images from PDF pages"""
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
        print(f"Extracted {len(image_list)} page images from PDF")
        return image_list
    except Exception as e:
        print(f"Error extracting images: {str(e)}")
        return []

def parse_manual_with_gemini(text):
    """Use Gemini AI to parse manual content"""
    if not GEMINI_API_KEY:
        print("Warning: No GEMINI_API_KEY. Using basic parsing.")
        return parse_manual_basic(text)
    
    try:
        prompt = f"""You are an expert at parsing assembly instruction manuals. Analyze this manual text and extract:

1. PARTS LIST - All components, hardware, and tools needed
2. ASSEMBLY STEPS - Step-by-step instructions in order

Manual Text:
{text[:8000]}

Return ONLY valid JSON in this exact format:
{{
  "parts": [
    {{"id": "A", "name": "Long screw", "quantity": 4}},
    {{"id": "B", "name": "Wooden panel", "quantity": 2}}
  ],
  "steps": [
    {{"number": 1, "instruction": "Attach panel A to base using screws B"}},
    {{"number": 2, "instruction": "Secure left side panel with brackets"}}
  ]
}}

Rules:
- Extract ALL parts mentioned
- Number steps sequentially from 1
- Keep instructions clear and concise
- If no parts/steps found, return empty arrays"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048
            }
        }
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={GEMINI_API_KEY}",
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
        else:
            print(f"Gemini API Error: {response.status_code}")
            
    except Exception as e:
        print(f"Gemini parsing failed: {str(e)}")
    
    return parse_manual_basic(text)

def parse_manual_basic(text):
    """Basic regex-based parsing fallback"""
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
    
    # Remove duplicates
    parts_list = list({p['name']: p for p in parts_list}.values())
    steps = sorted(steps, key=lambda x: x['number'])
    
    print(f"Basic parsing: {len(parts_list)} parts, {len(steps)} steps")
    return parts_list, steps

@app.post("/upload-manual")
async def upload_manual(file: UploadFile = File(...)):
    """Upload and process assembly manual PDF"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing manual: {file.filename}")
        print(f"{'='*60}")
        
        # Read PDF
        pdf_content = await file.read()
        pdf_file = BytesIO(pdf_content)
        
        # Extract text
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_file)
        print(f"First 500 chars: {text[:500]}\n")
        
        # Extract images
        print("Extracting page images...")
        pdf_file.seek(0)
        page_images = extract_images_from_pdf(pdf_content)
        
        # Parse with AI
        print("Parsing manual with AI...")
        parts_list, steps = parse_manual_with_gemini(text)
        
        # Store in global context
        manual_context["text"] = text
        manual_context["parts_list"] = parts_list
        manual_context["steps"] = steps
        manual_context["current_step"] = 0
        manual_context["images"] = page_images
        
        print(f"\n{'='*60}")
        print(f"✓ Manual processed successfully!")
        print(f"  Parts found: {len(parts_list)}")
        print(f"  Steps found: {len(steps)}")
        print(f"  Pages: {len(page_images)}")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "content": f"Manual uploaded: {len(parts_list)} parts, {len(steps)} steps",
            "parts_count": len(parts_list),
            "steps_count": len(steps),
            "parts": parts_list[:10],
            "steps": steps[:5],
            "first_step": steps[0] if steps else None,
            "debug_info": {
                "text_length": len(text),
                "pages_analyzed": len(page_images),
                "parsing_method": "gemini" if GEMINI_API_KEY else "basic"
            }
        }
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Error processing manual: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "parts_count": 0,
            "steps_count": 0
        }

@app.post("/next-step")
def next_step():
    """Move to next assembly step"""
    steps = manual_context.get("steps", [])
    current = manual_context.get("current_step", 0)
    
    if current < len(steps) - 1:
        manual_context["current_step"] = current + 1
        next_step_data = steps[current + 1]
        
        return {
            "success": True,
            "step": next_step_data,
            "step_number": current + 1,
            "total_steps": len(steps)
        }
    else:
        return {
            "success": False,
            "message": "Already at final step"
        }

@app.post("/previous-step")
def previous_step():
    """Move to previous assembly step"""
    current = manual_context.get("current_step", 0)
    
    if current > 0:
        manual_context["current_step"] = current - 1
        steps = manual_context.get("steps", [])
        prev_step_data = steps[current - 1]
        
        return {
            "success": True,
            "step": prev_step_data,
            "step_number": current - 1,
            "total_steps": len(steps)
        }
    else:
        return {
            "success": False,
            "message": "Already at first step"
        }

@app.get("/manual-info")
def get_manual_info():
    """Get current manual information"""
    steps = manual_context.get("steps", [])
    current_step_num = manual_context.get("current_step", 0)
    
    return {
        "has_manual": bool(manual_context.get("text")),
        "parts_count": len(manual_context.get("parts_list", [])),
        "steps_count": len(steps),
        "current_step": current_step_num,
        "parts": manual_context.get("parts_list", [])[:10],
        "current_step_info": steps[current_step_num] if steps and current_step_num < len(steps) else None,
        "all_steps": steps
    }

@app.get("/stats")
def get_stats():
    """Get performance statistics"""
    return {
        "total_detections": detection_stats["total_requests"],
        "avg_response_time_ms": int(detection_stats["avg_response_time"] * 1000),
        "last_detection_time_ms": int(detection_stats["last_detection_time"] * 1000),
        "manual_loaded": bool(manual_context.get("text")),
        "current_step": manual_context.get("current_step", 0),
        "total_steps": len(manual_context.get("steps", []))
    }

@app.get("/api-status")
def api_status():
    """Check API configuration status"""
    return {
        "huggingface_configured": bool(HF_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "yolo_loaded": yolo_model is not None,
        "manual_uploaded": bool(manual_context.get("text")),
        "parsing_capabilities": {
            "ai_parsing": bool(GEMINI_API_KEY),
            "visual_analysis": bool(HF_API_KEY),
            "basic_parsing": True
        },
        "performance": detection_stats
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "model_loaded": yolo_model is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "="*60)
    print("Sophie Vision-Guide Backend - Real-Time Detection")
    print("="*60)
    print("API Configuration:")
    print(f"  Hugging Face: {'✓' if HF_API_KEY else '✗'}")
    print(f"  Gemini AI: {'✓' if GEMINI_API_KEY else '✗'}")
    print(f"  YOLO: ✓ Loaded")
    print("\nOptimizations:")
    print("  • Fast YOLO inference")
    print("  • Automatic frame resizing")
    print("  • Low latency mode")
    print("  • Manual context caching")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)