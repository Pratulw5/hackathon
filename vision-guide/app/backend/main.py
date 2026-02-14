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

load_dotenv()
app = FastAPI()
import os
from dotenv import load_dotenv

load_dotenv()

print("HF_API_KEY:", os.getenv("HF_API_KEY"))
print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ai-vision-guide.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once at startup
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

# Hugging Face API configuration
HF_API_KEY = os.getenv("HF_API_KEY", "")
FLORENCE_API_URL = "https://api-inference.huggingface.co/models/microsoft/Florence-2-large"

# Use Gemini for intelligent text parsing (FREE tier available)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent"

# Global state to store manual content
manual_context = {
    "text": "",
    "parts_list": [],
    "steps": [],
    "current_step": 0,
    "images": []
}


class ImageData(BaseModel):
    image: str


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF manual - improved version"""
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
    """Extract images from PDF pages using pdf2image"""
    try:
        # Convert PDF pages to images
        images = convert_from_bytes(pdf_bytes, dpi=150, fmt='jpeg')
        
        image_list = []
        for i, img in enumerate(images):
            # Convert PIL Image to base64
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
    """Use Gemini AI to intelligently parse manual content"""
    
    if not GEMINI_API_KEY:
        print("Warning: No GEMINI_API_KEY set. Falling back to basic parsing.")
        return parse_manual_basic(text)
    
    try:
        prompt = f"""
You are an expert at parsing assembly instruction manuals. Analyze this manual text and extract:

1. PARTS LIST - All components, hardware, and tools needed
2. ASSEMBLY STEPS - Step-by-step instructions in order

Manual Text:
{text[:8000]}  # Limit to avoid token limits

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
- Extract ALL parts mentioned, even if not in a formal parts list
- Number steps sequentially starting from 1
- Keep instructions clear and concise (under 200 chars)
- If no parts/steps found, return empty arrays
"""

        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048
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
            
            # Extract the generated text
            if 'candidates' in result and len(result['candidates']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                
                # Clean up markdown code blocks if present
                generated_text = generated_text.replace('```json', '').replace('```', '').strip()
                
                # Parse JSON
                parsed_data = json.loads(generated_text)
                
                parts_list = parsed_data.get('parts', [])
                steps = parsed_data.get('steps', [])
                
                print(f"Gemini parsed: {len(parts_list)} parts, {len(steps)} steps")
                
                return parts_list, steps
        else:
            print(f"Gemini API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Gemini parsing failed: {str(e)}")
    
    # Fallback to basic parsing
    return parse_manual_basic(text)


def parse_manual_basic(text):
    """Improved basic parsing with better pattern matching"""
    parts_list = []
    steps = []
    
    if not text or len(text.strip()) < 50:
        print("Text too short or empty")
        return parts_list, steps
    
    # More flexible parts extraction
    # Look for various patterns
    patterns = [
        # Pattern 1: Letter/Number followed by dash or colon and description
        r'([A-Z]\d*|\d+[A-Z]?)\s*[-:\.]\s*([^\n]+?)(?:\s*\((\d+)\s*(?:pcs?|pieces?)?\))?',
        # Pattern 2: Bullet points or numbered items
        r'[•\-\*]\s*([^\n:]+?):\s*([^\n]+?)(?:\s*\((\d+)\s*(?:pcs?|pieces?)?\))?',
        # Pattern 3: Just item descriptions with quantities
        r'(\d+)\s*x\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if len(match) >= 2:
                part_id = match[0].strip() if match[0] else f"P{len(parts_list)+1}"
                part_name = match[1].strip()
                quantity = match[2].strip() if len(match) > 2 and match[2] else "1"
                
                if len(part_name) > 3:  # Valid part name
                    parts_list.append({
                        "id": part_id,
                        "name": part_name[:80],
                        "quantity": quantity
                    })
    
    # Extract steps with multiple patterns
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
            
            if len(instruction) > 10:  # Valid instruction
                steps.append({
                    "number": step_num,
                    "instruction": instruction[:250]
                })
    
    # Remove duplicates and sort
    parts_list = list({p['name']: p for p in parts_list}.values())
    steps = sorted(steps, key=lambda x: x['number'])
    
    # If still nothing found, try to extract anything that looks relevant
    if not parts_list and not steps:
        # Look for any numbered or bulleted lists
        lines = text.split('\n')
        current_step = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip empty or very short lines
            if len(line) < 10:
                continue
            
            # Check if it's a step-like instruction
            if any(keyword in line.lower() for keyword in ['attach', 'insert', 'place', 'connect', 'secure', 'install']):
                current_step += 1
                steps.append({
                    "number": current_step,
                    "instruction": line[:250]
                })
            
            # Check if it's a part-like item
            elif any(keyword in line.lower() for keyword in ['screw', 'bolt', 'panel', 'bracket', 'piece', 'part']):
                parts_list.append({
                    "id": f"P{len(parts_list)+1}",
                    "name": line[:80],
                    "quantity": "1"
                })
    
    print(f"Basic parsing found: {len(parts_list)} parts, {len(steps)} steps")
    
    return parts_list, steps


def analyze_pdf_page_with_florence(image_base64):
    """Use Florence-2 to extract information from PDF page images"""
    
    if not HF_API_KEY:
        return None
    
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        # Extract base64 data
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
        
        image_bytes = base64.b64decode(image_base64)
        
        # Use Florence-2 for detailed captioning
        prompt = "<DETAILED_CAPTION>"
        
        response = requests.post(
            FLORENCE_API_URL,
            headers=headers,
            files={"file": image_bytes},
            data={"inputs": prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Florence-2 analysis: {result}")
            return result
        
    except Exception as e:
        print(f"Florence-2 analysis failed: {str(e)}")
    
    return None


@app.post("/upload-manual")
async def upload_manual(file: UploadFile = File(...)):
    """Upload and process assembly manual PDF with AI-powered extraction"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing manual: {file.filename}")
        print(f"{'='*60}")
        
        # Read PDF file
        pdf_content = await file.read()
        pdf_file = BytesIO(pdf_content)
        
        # Step 1: Extract text from PDF
        print("Step 1: Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_file)
        
        # Debug: Show first 500 chars
        print(f"First 500 chars of extracted text:\n{text[:500]}\n")
        
        # Step 2: Extract images from PDF pages (for visual analysis)
        print("Step 2: Extracting page images...")
        pdf_file.seek(0)  # Reset file pointer
        page_images = extract_images_from_pdf(pdf_content)
        
        # Step 3: Use AI to parse the manual
        print("Step 3: Parsing manual with AI...")
        parts_list, steps = parse_manual_with_gemini(text)
        
        # Step 4: If AI parsing failed or found nothing, try enhanced visual analysis
        if (not parts_list or not steps) and page_images:
            print("Step 4: Trying visual analysis with Florence-2...")
            
            # Analyze first few pages with Florence-2
            for img_data in page_images[:3]:  # Analyze first 3 pages
                florence_result = analyze_pdf_page_with_florence(img_data['data'])
                
                if florence_result:
                    # Extract any additional information from visual analysis
                    # This could help find parts lists in images/diagrams
                    print(f"Visual analysis of page {img_data['page']}: {florence_result}")
        
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
        
        # Return detailed response
        return {
            "success": True,
            "content": f"Manual uploaded: {len(parts_list)} parts identified, {len(steps)} assembly steps found",
            "parts_count": len(parts_list),
            "steps_count": len(steps),
            "parts": parts_list[:10],  # Return first 10 parts
            "steps": steps[:5],  # Return first 5 steps
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
    """Manual-aware detection using Florence-2 + manual context"""
    
    try:
        # Get current step instruction
        current_step = manual_context.get("current_step", 0)
        steps = manual_context.get("steps", [])
        parts = manual_context.get("parts_list", [])
        
        current_instruction = ""
        if steps and current_step < len(steps):
            current_instruction = steps[current_step]["instruction"]
        
        # Create intelligent prompt based on manual context
        if current_instruction:
            prompt = f"Identify and locate these items for assembly: {current_instruction[:100]}"
        elif parts:
            # Use parts list to create prompt
            part_names = [p['name'] for p in parts[:5]]
            prompt = f"Locate these assembly parts: {', '.join(part_names)}"
        else:
            prompt = "Identify all tools and parts in this assembly workspace"
        
        print(f"Detection prompt: {prompt}")
        
        # Run standard YOLO detection first (fast)
        yolo_result = detect(data)
        detections = yolo_result.get("detections", [])
        
        # Enhance with Florence-2 if we have manual context
        if HF_API_KEY and (parts or steps):
            try:
                florence_result = analyze_with_huggingface_api(data.image, prompt)
                florence_detections = parse_florence_output(florence_result)
                
                # Merge detections (prefer Florence-2 for manual mode)
                if florence_detections:
                    detections = florence_detections
                    print(f"Using Florence-2 detections: {len(detections)}")
            except Exception as e:
                print(f"Florence-2 enhancement failed, using YOLO: {str(e)}")
        
        # Generate instruction based on detections and manual
        instruction = ""
        if current_instruction and detections:
            detected_labels = [d['label'] for d in detections[:3]]
            instruction = f"Step {current_step + 1}: {current_instruction[:100]}"
            if detected_labels:
                instruction += f" (Detected: {', '.join(detected_labels)})"
        elif detections:
            labels = [d['label'] for d in detections[:3]]
            instruction = f"I found {len(detections)} items: {', '.join(labels)}"
        else:
            instruction = "Point camera at the parts for the current step"
        
        return {
            "detections": detections,
            "instruction": instruction,
            "current_step": current_step,
            "total_steps": len(steps),
            "current_step_detail": steps[current_step] if steps and current_step < len(steps) else None
        }
        
    except Exception as e:
        print(f"Error in manual-based detection: {str(e)}")
        # Fallback to YOLO
        return detect(data)


def analyze_with_huggingface_api(image_base64, prompt):
    """Call Hugging Face Inference API for Florence-2"""
    
    if not HF_API_KEY:
        return None
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    if "base64," in image_base64:
        image_base64 = image_base64.split("base64,")[1]
    
    image_bytes = base64.b64decode(image_base64)
    
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
            print(f"Florence API Error: {response.status_code}")
            
    except Exception as e:
        print(f"Florence API call failed: {str(e)}")
    
    return None


def parse_florence_output(result):
    """Parse Florence-2 API output into detections"""
    detections = []
    
    if not result:
        return detections
    
    try:
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        if isinstance(result, dict):
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
                            "color": "#a855f7",
                            "highlight": True,
                            "confidence": 0.95
                        })
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
        }
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    print("\n" + "="*60)
    print("Sophie Vision-Guide Backend - AI-Powered Manual Processing")
    print("="*60)
    print("API Configuration:")
    print(f"  Hugging Face (Florence-2): {'✓ Configured' if HF_API_KEY else '✗ Not configured'}")
    print(f"  Google Gemini (Text AI):  {'✓ Configured' if GEMINI_API_KEY else '✗ Not configured'}")
    print(f"  YOLO Object Detection:    ✓ Loaded")
    print("\nParsing Capabilities:")
    print(f"  AI-Powered Parsing:  {'✓ Enabled' if GEMINI_API_KEY else '✗ Disabled (using basic)'}")
    print(f"  Visual Analysis:     {'✓ Enabled' if HF_API_KEY else '✗ Disabled'}")
    print(f"  Basic Pattern Match: ✓ Always available")
    print("="*60)
    print("\nTo enable AI parsing:")
    print("  1. Get Gemini API key: https://makersuite.google.com/app/apikey")
    print("  2. Set: export GEMINI_API_KEY='your_key_here'")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)