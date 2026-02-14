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
# Add at top of hosted-backend.py
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",  "https://ai-vision-guide.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once at startup
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")  # nano model (fast)

# Hugging Face API configuration
# Get your free API key from: https://huggingface.co/settings/tokens
HF_API_KEY = os.getenv("HF_API_KEY", "")  # Set this in environment
FLORENCE_API_URL = "https://api-inference.huggingface.co/models/microsoft/Florence-2-large"

# Alternative: Use Replicate API (also free tier)
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", "")

# Global state to store manual content
manual_context = {
    "text": "",
    "parts_list": [],
    "steps": [],
    "current_step": 0
}


class ImageData(BaseModel):
    image: str


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF manual"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def parse_manual_content(text):
    """Parse manual text to extract parts and steps"""
    parts_list = []
    steps = []
    
    # Extract parts list (look for common patterns)
    parts_section = re.search(r'(PARTS LIST|PARTS|COMPONENTS|HARDWARE)(.*?)(ASSEMBLY|INSTRUCTIONS|STEPS)', 
                              text, re.IGNORECASE | re.DOTALL)
    if parts_section:
        parts_text = parts_section.group(2)
        # Extract items that look like parts (letter/number followed by description)
        parts = re.findall(r'([A-Z]\d*|\d+[A-Z]?)[:\.\-\s]+(.*?)(?:\n|$)', parts_text)
        for part_id, description in parts:
            parts_list.append({
                "id": part_id.strip(),
                "name": description.strip()[:50]  # Limit length
            })
    
    # Extract assembly steps
    steps_section = re.search(r'(ASSEMBLY|INSTRUCTIONS|STEPS)(.*)', 
                              text, re.IGNORECASE | re.DOTALL)
    if steps_section:
        steps_text = steps_section.group(2)
        # Extract numbered steps
        step_matches = re.findall(r'(?:STEP\s+)?(\d+)[:\.\-\s]+(.*?)(?=(?:STEP\s+)?\d+[:\.\-]|$)', 
                                  steps_text, re.IGNORECASE | re.DOTALL)
        for step_num, step_text in step_matches:
            steps.append({
                "number": int(step_num),
                "instruction": step_text.strip()[:200]  # Limit length
            })
    
    return parts_list, steps


def analyze_with_huggingface_api(image_base64, prompt):
    """Call Hugging Face Inference API for Florence-2"""
    
    if not HF_API_KEY:
        print("Warning: No HF_API_KEY set. Using public API (rate limited)")
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    
    # Convert base64 to bytes
    if "base64," in image_base64:
        image_base64 = image_base64.split("base64,")[1]
    
    image_bytes = base64.b64decode(image_base64)
    
    # Prepare request
    try:
        response = requests.post(
            FLORENCE_API_URL,
            headers=headers,
            files={"file": image_bytes},
            data={"inputs": prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return None


def analyze_with_replicate_api(image_base64, prompt):
    """Call Replicate API for Florence-2 (alternative)"""
    
    if not REPLICATE_API_KEY:
        print("Warning: No REPLICATE_API_KEY set")
        return None
    
    try:
        import replicate
        
        # Run Florence-2 on Replicate
        output = replicate.run(
            "adirik/florence-2:5f19c6e967d1e5b16f224c9e2c0d42c0bb8859f4a7fe8bf27c4be0c5dfb2a5cf",
            input={
                "image": image_base64,
                "task": "caption_to_phrase_grounding",
                "text": prompt
            }
        )
        
        return output
        
    except Exception as e:
        print(f"Replicate API call failed: {str(e)}")
        return None


def parse_florence_output(result, task_type="grounding"):
    """Parse Florence-2 API output into detections"""
    detections = []
    
    if not result:
        return detections
    
    try:
        # Hugging Face API returns different formats
        # Handle different response structures
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        if isinstance(result, dict):
            # Look for bounding boxes and labels
            if 'bboxes' in result and 'labels' in result:
                bboxes = result['bboxes']
                labels = result['labels']
                
                for bbox, label in zip(bboxes, labels):
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        detections.append({
                            "label": str(label),
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(x2 - x1),
                            "height": float(y2 - y1),
                            "color": "#a855f7",  # Purple for manual-based
                            "highlight": True,
                            "confidence": 0.95
                        })
            
            # Alternative format: regions with descriptions
            elif 'regions' in result:
                for region in result['regions']:
                    if 'bbox' in region:
                        bbox = region['bbox']
                        label = region.get('label', region.get('description', 'object'))
                        
                        detections.append({
                            "label": label,
                            "x": float(bbox[0]),
                            "y": float(bbox[1]),
                            "width": float(bbox[2] - bbox[0]),
                            "height": float(bbox[3] - bbox[1]),
                            "color": "#06b6d4",
                            "confidence": 0.90
                        })
        
    except Exception as e:
        print(f"Error parsing Florence output: {str(e)}")
    
    return detections


@app.post("/upload-manual")
async def upload_manual(file: UploadFile = File(...)):
    """Upload and process assembly manual PDF"""
    try:
        # Read PDF file
        pdf_content = await file.read()
        pdf_file = BytesIO(pdf_content)
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        
        # Parse manual to extract parts and steps
        parts_list, steps = parse_manual_content(text)
        
        # Store in global context
        manual_context["text"] = text
        manual_context["parts_list"] = parts_list
        manual_context["steps"] = steps
        manual_context["current_step"] = 0
        
        print(f"Manual processed: {len(parts_list)} parts, {len(steps)} steps")
        
        return {
            "success": True,
            "content": f"Manual uploaded: {len(parts_list)} parts identified, {len(steps)} assembly steps found",
            "parts_count": len(parts_list),
            "steps_count": len(steps),
            "parts": parts_list[:5],  # Return first 5 parts
            "first_step": steps[0] if steps else None
        }
        
    except Exception as e:
        print(f"Error processing manual: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/detect")
def detect(data: ImageData):
    """Standard YOLO object detection"""
    
    # Decode base64 image
    img_bytes = base64.b64decode(data.image.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = yolo_model(frame)

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
    
    print(f"YOLO detected {len(detections)} objects")
    return {"detections": detections}


@app.post("/detect-with-manual")
def detect_with_manual(data: ImageData):
    """Manual-aware detection using hosted Florence-2 API"""
    
    try:
        # Get current step instruction
        current_step = manual_context.get("current_step", 0)
        steps = manual_context.get("steps", [])
        
        current_instruction = ""
        if steps and current_step < len(steps):
            current_instruction = steps[current_step]["instruction"]
        
        # Create prompt based on manual context
        if current_instruction:
            # Extract key objects from instruction
            # Simple keyword extraction (can be improved with NLP)
            keywords = re.findall(r'\b(screw|panel|bracket|bolt|nut|hole|base|part|piece|connector)\w*\b', 
                                 current_instruction.lower())
            
            if keywords:
                prompt = f"Locate and identify: {', '.join(set(keywords[:5]))}"
            else:
                prompt = f"Identify all parts and tools in this assembly workspace"
        else:
            prompt = "Identify all tools and parts in the image"
        
        print(f"Florence-2 prompt: {prompt}")
        
        # Call hosted API
        # Try Hugging Face first, fallback to Replicate
        florence_result = analyze_with_huggingface_api(data.image, prompt)
        
        if not florence_result:
            print("HuggingFace API failed, trying Replicate...")
            florence_result = analyze_with_replicate_api(data.image, prompt)
        
        # Parse results
        detections = parse_florence_output(florence_result)
        
        # If Florence API fails, fallback to YOLO
        if not detections:
            print("Florence-2 API failed, falling back to YOLO")
            return detect(data)
        
        # Generate instruction based on detections
        instruction = ""
        if current_instruction and detections:
            detected_labels = [d['label'] for d in detections]
            instruction = f"Step {current_step + 1}: {current_instruction[:100]}... I can see: {', '.join(detected_labels[:3])}"
        elif detections:
            instruction = f"I detected: {', '.join([d['label'] for d in detections[:3]])}"
        else:
            instruction = "Point camera at the parts mentioned in the manual"
        
        print(f"Florence-2 API detected {len(detections)} objects")
        
        return {
            "detections": detections,
            "instruction": instruction,
            "current_step": current_step,
            "total_steps": len(steps)
        }
        
    except Exception as e:
        print(f"Error in manual-based detection: {str(e)}")
        # Fallback to YOLO
        return detect(data)


@app.post("/next-step")
def next_step():
    """Move to next assembly step"""
    steps = manual_context.get("steps", [])
    current = manual_context.get("current_step", 0)
    
    if current < len(steps) - 1:
        manual_context["current_step"] = current + 1
        next_step = steps[current + 1]
        return {
            "success": True,
            "step": next_step,
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
        prev_step = steps[current - 1]
        return {
            "success": True,
            "step": prev_step,
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
    return {
        "has_manual": bool(manual_context.get("text")),
        "parts_count": len(manual_context.get("parts_list", [])),
        "steps_count": len(manual_context.get("steps", [])),
        "current_step": manual_context.get("current_step", 0),
        "parts": manual_context.get("parts_list", [])[:10],
        "current_step_info": manual_context.get("steps", [])[manual_context.get("current_step", 0)] if manual_context.get("steps") else None
    }


@app.get("/api-status")
def api_status():
    """Check API configuration status"""
    return {
        "huggingface_configured": bool(HF_API_KEY),
        "replicate_configured": bool(REPLICATE_API_KEY),
        "yolo_loaded": yolo_model is not None,
        "manual_uploaded": bool(manual_context.get("text"))
    }


if __name__ == "__main__":
    import uvicorn

    # Get Render-assigned port or fallback to 8000
    port = int(os.environ.get("PORT", 8000))

    # Print API key status
    print("\n" + "="*50)
    print("API Configuration Status:")
    print(f"  Hugging Face API: {'✓ Configured' if HF_API_KEY else '✗ Not configured (using public API - rate limited)'}")
    print(f"  Replicate API: {'✓ Configured' if REPLICATE_API_KEY else '✗ Not configured'}")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
