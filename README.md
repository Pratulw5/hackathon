# Sophiie AI Agents Hackathon 2026

**Build the future of AI-human interaction.**

| | |
|---|---|
| **What** | A solo hackathon focused on AI agent interaction — voice, text, UX, and UI |
| **When** | February 14–15, 2026 (Saturday–Sunday) |
| **Where** | Virtual — participate from anywhere in Australia |
| **Prize** | **$5,000 AUD cash** (1st place) + job offers for top performers |
| **Format** | Solo only — show us what *you* can build |
| **Hacking Time** | 33 hours |

---

## The Challenge

**Design and build an AI agent with an exceptional interaction experience.**

We want to see how you think about the space between humans and AI. This is deliberately open-ended — you choose the problem, the modality, and the approach. What matters is the *interaction*.

Some directions to inspire you (not requirements):

- A voice agent that feels natural to talk to
- A text-based assistant with a thoughtful, intuitive UX
- A multi-modal agent that blends voice, text, and visual elements
- An agent that handles a complex workflow through conversation
- Something we haven't thought of yet

**You will be judged on innovation, technical execution, and how good the interaction feels** — not just whether the AI works, but whether a human would *want* to use it.

Use any tech stack. Use any AI provider. Use AI coding assistants. The only constraint is time.

---

## Schedule

All times are **AEST (Australian Eastern Standard Time, UTC+10 — Brisbane time)**.

### Saturday, February 14

| Time | Event |
|------|-------|
| **9:00 AM** | Kickoff — challenge explained, rules confirmed |
| **9:30 AM** | **Hacking begins** |
| 12:00 PM | Office hours / Q&A (optional, Discord) |
| 4:00 PM | Community check-in / progress sharing (optional, Discord) |

### Sunday, February 15

| Time | Event |
|------|-------|
| **6:00 PM** | **Submission deadline — hard cut-off, no exceptions** |

### After the Hackathon

| When | Event |
|------|-------|
| Feb 16 – Feb 28 | Judging period — judges review all submissions |
| ~Early March | Winners announced via livestream (details shared on Discord and Email) |

---

## Rules

### The Essentials

1. **Solo only** — one person per submission, no teams
2. **No pre-work** — all project code must be written during the hackathon window (after 9:30 AM AEST, Feb 14)
3. **Public GitHub repo** — your repository must be publicly visible at time of submission
4. **AI assistance is allowed** — Copilot, Claude, ChatGPT, Cursor, whatever you want. You still need to build it within the timeframe
5. **Must be functional** — your project must run and be demonstrable, not just a concept or slide deck
6. **One submission per person** — you may iterate, but submit one final project

### What You CAN Prepare Before Kickoff

- Research, planning, and brainstorming (on paper, in your head — just not in code)
- Setting up your development environment
- Reading documentation for tools/APIs you plan to use
- Creating accounts (GitHub, API providers, etc.)
- Watching tutorials

### What You CANNOT Do Before Kickoff

- Write any project code
- Create your project repository
- Fork/clone an existing project and modify it
- Build components, libraries, or templates specifically for your submission
- Start a project in a private repo then make it public later

### How We Verify

We will check:
- **Repository creation date** — must be after 9:30 AM AEST, Feb 14
- **Commit history** — should show natural progression, not a single massive commit
- **First commit timestamp** — must be after kickoff

**Red flags that will result in disqualification:**
- Repo created before the hackathon
- Single commit containing the entire project
- Commits timestamped before kickoff
- Evidence of code copied from a pre-existing private repo

---

## Submission Requirements

**Deadline: 6:00 PM AEST, Sunday February 15, 2026 — hard cut-off.**

To submit, you must complete **all** of the following:

1. **Public GitHub repo** — created after kickoff, with a clear commit history
2. **This README** — fill out the [Your Submission](#your-submission) section below
3. **Demo video** (2–5 minutes) — show your agent in action, explain your approach
4. **Working project** — judges must be able to understand and evaluate your agent from the repo + video

### How to Submit

1. Fork this repository
2. Build your project in the fork
3. Fill out the [Your Submission](#your-submission) section below
4. Record your demo video and add the link to your submission
5. Ensure your repo is **public** before 6:00 PM AEST Sunday
6. Submit your repo link via the submission form (link will be shared at kickoff)

---

## Judging Criteria

| Criteria | Weight | What We're Looking For |
|----------|--------|----------------------|
| **Interaction Design** | 30% | How intuitive, natural, and delightful is the human-AI interaction? Does it feel good to use? |
| **Innovation** | 25% | Novel approach, creative problem-solving, or a fresh take on agent interaction |
| **Technical Execution** | 25% | Code quality, architecture, reliability, completeness |
| **Presentation** | 20% | Demo quality, clarity of communication, ability to convey your vision |

### Judges

Sophiie senior engineers and CTO. Judging will take place over a 2-week period following the submission deadline.

---

## Prizes

| Place | Prize |
|-------|-------|
| **1st Place** | **$5,000 AUD cash** |
| **Top Performers** | Job offers or interview fast-tracks at Sophiie* |
| **All Finalists** | Consideration for current and future roles |

*\*Job offers and interview fast-tracks are entirely at the discretion of Sophiie and are not guaranteed.*

> Participants retain full ownership and IP of their submissions. Sophiie receives a non-exclusive license to review and evaluate submissions for judging purposes only.

---

## Your Submission

> **Instructions:** Fill out this section in your forked repo. This is what judges will see first.

### Participant

| Field | Your Answer |
|-------|-------------|
| **Name** |Pratul Wadhwa |
| **University / Employer** | University Of Queensland / Distrosub|

### Project

| Field | Your Answer |
|-------|-------------|
| **Project Name** | Vision Guide - AI Assembly Assistant |
| **One-Line Description** | Real-time AR voice assistant that guides users through product assembly using computer vision, manual parsing, and voice interaction |
| **Demo Video Link** | https://www.awesomescreenshot.com/video/49458079?key=3dea16d84bc3b8cc25ac7a260c7d4d7e |
| **Tech Stack** | Next.js, FastAPI, YOLOv8, Claude Sonnet 4.5, Faster-Whisper, gTTS, Tailwind CSS, OpenCV |
| **AI Provider(s) Used** | Anthropic (Claude Sonnet 4.5 for manual parsing & voice Q&A), Whisper (speech-to-text), gTTS/edge-tts (text-to-speech) |

### About Your Project

#### What does it do?

Vision Guide transforms traditional PDF manuals into an interactive AR-powered voice assistant. Users upload their assembly manual, and the AI instantly parses it to extract parts lists and step-by-step instructions. The app then activates a live camera feed with real-time object detection, using YOLOv8 to identify components in the user's workspace.

As users assemble their product, they can speak naturally to ask questions like "where is the charging port?" or "what's the next step?" The AI assistant responds with voice guidance while highlighting relevant objects on screen with AR overlays. The system understands the manual's context, tracks progress through steps, and provides encouragement along the way.

The tutorial mode walks users through each step sequentially, detecting required parts and confirming completion before moving forward. This hands-free, eyes-on-the-work approach makes complex assembly tasks accessible to everyone, from IKEA furniture to laptop repairs.

#### How does the interaction work?

When users first open the app, they're greeted with a camera view and prompted to upload their PDF manual. The AI immediately processes the document, extracting all parts and instructions using Claude's vision capabilities. Once processed, voice recognition activates automatically.

Users simply speak commands like "start tutorial" to begin guided assembly. The camera continuously scans their workspace, drawing colored boxes around detected objects. When the AI speaks instructions, voice recognition pauses automatically to avoid picking up its own voice, then resumes listening seamlessly.

The interface shows real-time detection results, current step progress, and visual overlays highlighting required components. Users can ask questions at any time ("which screw do I need?"), mark steps complete, or jump to specific instructions. Everything is hands-free - no need to touch the screen with dirty or busy hands during assembly.

#### What makes it special?

**Smart Context Awareness**: Unlike generic voice assistants, Vision Guide understands the specific manual you're working with. It knows every part, every step, and can see what's in front of you through real-time object detection.

**Seamless Voice Control**: The speech recognition intelligently pauses during AI responses to prevent echo, then auto-resumes - creating a natural conversation flow that feels like having an expert guide standing next to you.

**Progressive Assembly Tracking**: The system doesn't just read instructions - it actively tracks which parts you need for each step, highlights them in your camera view, and confirms completion before advancing. This reduces errors and confusion.

**100% Free Open-Source Stack**: Built entirely with free tools (Faster-Whisper for STT, gTTS for TTS, Claude API, YOLOv8), making professional AR assembly assistance accessible to everyone without expensive APIs or hardware requirements.

**Real Production Value**: This isn't a prototype - it's a polished, deployable application with error handling, multiple fallbacks for each service, proper audio conversion, and a beautiful UI that works on mobile and desktop.

#### How to run it
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/vision-guide
cd vision-guide

# 2. Install frontend dependencies
cd frontend
npm install

# 3. Set up frontend environment
cp .env.example .env.local
# Add your backend URL:
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# 4. Install backend dependencies
cd ../backend
pip install -r requirements.txt

# Required packages:
pip install fastapi uvicorn ultralytics opencv-python PyPDF2 pdf2image
pip install python-dotenv gtts faster-whisper pydub

# 5. Set up backend environment
cp .env.example .env
# Add your API keys:
GEMINI_API_KEY=your_gemini_key_here
# (Optional: SUPABASE_URL, SUPABASE_KEY for persistence)

# 6. Install ffmpeg (required for audio conversion)
# Windows: Download from https://ffmpeg.org/download.html
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# 7. Start the backend
python main.py
# Backend runs on http://localhost:8000

# 8. Start the frontend (new terminal)
cd ../frontend
npm run dev
# Frontend runs on http://localhost:3000

# 9. Grant permissions
# Allow camera and microphone access when prompted by browser

# 10. Upload a manual and start assembling!
```

**Quick test**: Visit http://localhost:3000, upload any PDF manual, click "ACTIVATE" voice button, and say "start tutorial"

#### Architecture / Technical Notes

**Frontend Architecture**: Built with Next.js 14 App Router using TypeScript and Tailwind CSS. Uses MediaRecorder API to capture 3-second audio chunks, automatically sending them to the backend STT endpoint. Implements proper state management with useRef for voice recognition lifecycle, preventing double-start errors and handling auto-restart after AI speech.

**Backend Architecture**: FastAPI server with four main pipelines:
1. **Manual Processing**: PyPDF2 extracts text → Claude Sonnet 4.5 parses into structured JSON → Optional Supabase persistence
2. **Object Detection**: YOLOv8s runs on each frame → Filters detections based on current step's required objects → Returns with AR overlay coordinates
3. **Speech-to-Text**: WebM audio → pydub converts to WAV → Faster-Whisper transcribes → Returns text command
4. **Voice Assistant**: Question + manual context + live detections → Claude generates response → gTTS converts to speech

**Key Technical Decisions**:
- **Audio Format Handling**: Implemented WebM-to-WAV conversion because Faster-Whisper can't parse Opus codec directly. Used pydub + ffmpeg pipeline.
- **Voice Loop Management**: STT endpoint stops voice recognition before playing audio, resumes after playback using audio.onended callback. Prevents AI from hearing itself.
- **Context-Aware Detection**: Instead of showing all detected objects, filters based on manual's current step requirements using fuzzy matching on part keywords.
- **Stateless with Memory**: Backend stores manual context in-memory for speed, with optional Supabase persistence for production deployment.

**Interesting Implementation**: The tutorial mode uses refs (manualUploadedRef, stepsRef) to avoid React closure issues in voice recognition callbacks, ensuring the latest state is always accessible even though recognition runs continuously.

---

## Code of Conduct

All participants must adhere to a standard of respectful, professional behavior. Harassment, discrimination, or disruptive behavior of any kind will result in immediate disqualification.

By participating, you agree to:
- Treat all participants, judges, and organizers with respect
- Submit only your own original work created during the hackathon
- Not interfere with other participants' work
- Follow the rules outlined in this document
---

## Communication & Support

- **Discord** — join the hackathon Discord server for announcements, Q&A, and community chat (link provided upon registration)
- **Office hours** — available during the event for technical questions

---

## FAQ

**Q: Can I use boilerplate / starter templates?**
A: You can use publicly available boilerplate (e.g., `create-react-app`, `Next.js` starter) as a starting point. You cannot use custom templates you built specifically for this hackathon before kickoff.

**Q: Can I use existing open-source libraries and APIs?**
A: Yes. You can use any publicly available libraries, frameworks, APIs, and services. The code *you* write must be created during the hackathon.

**Q: Do I need to be in Australia?**
A: Preferred but not strictly required. The hackathon is primarily targeted at Australian residents and students, but we won't turn away great talent.

**Q: Can I use AI coding tools like Copilot or Claude?**
A: Absolutely. Use whatever tools you want. The 33-hour time constraint is the great equalizer.

**Q: What if I can't finish?**
A: Submit what you have. A well-thought-out partial project with a great demo video can still score well. We're evaluating your thinking and skill, not just completion.

**Q: How will I know if I won?**
A: Winners will be announced via livestream approximately 2 weeks after the hackathon. All participants will be notified.

**Q: Can I keep working on my project after the deadline?**
A: You can continue developing after the hackathon, but **only the state of your repo at 6:00 PM AEST Sunday Feb 15 will be judged**. We will check commit timestamps.

---

## About Sophiie

Sophiie is an AI office manager for trades businesses — helping plumbers, electricians, builders, and other trade professionals run their operations with intelligent automation. We're a team that cares deeply about how humans interact with AI, and we're looking for people who think the same way.

[sophiie.com](https://sophiie.com)

---

**Good luck. Build something that makes us say "wow."**
