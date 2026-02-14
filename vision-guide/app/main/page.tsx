'use client';

import { useState, useEffect, useRef } from 'react';

export default function CameraApp() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<any>(null);
  const animationFrameRef = useRef<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // State
  const [isLoading, setIsLoading] = useState(true);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [statusText, setStatusText] = useState('Ready');
  const [currentStep, setCurrentStep] = useState(0);
  const [overlaysActive, setOverlaysActive] = useState(false);
  const [instruction, setInstruction] = useState('');
  const [showInstruction, setShowInstruction] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [manualUploaded, setManualUploaded] = useState(false);
  const [manualContent, setManualContent] = useState<string>('');
  const [showManualPanel, setShowManualPanel] = useState(false);
  const [isProcessingManual, setIsProcessingManual] = useState(false);
  const [detectionMode, setDetectionMode] = useState<'yolo' | 'manual'>('yolo');

  useEffect(() => {
    drawOverlays();
  }, [detections]);

  // Test overlays data
  const testOverlays = [
    {
      type: 'circle',
      x: 0.3,
      y: 0.4,
      label: 'Insert screw here',
      color: '#06b6d4'
    },
    {
      type: 'circle',
      x: 0.7,
      y: 0.5,
      label: 'Attach panel',
      color: '#a855f7'
    },
    {
      type: 'arrow',
      startX: 0.2,
      startY: 0.3,
      endX: 0.3,
      endY: 0.4,
      color: '#f59e0b'
    }
  ];

  // Assembly steps
  const assemblySteps = [
    "Point your camera at the parts to begin",
    "Insert the screw into the marked hole",
    "Attach the left panel to the base",
    "Secure with the remaining screws",
    "Assembly complete! Great job!"
  ];

  // Initialize camera
  useEffect(() => {
    initCamera();
    setupVoiceRecognition();

    return () => {
      // Cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (recognitionRef.current && isListening) {
        recognitionRef.current.stop();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Canvas resize
  useEffect(() => {
    const resizeCanvas = () => {
      if (canvasRef.current) {
        canvasRef.current.width = window.innerWidth;
        canvasRef.current.height = window.innerHeight;
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    return () => window.removeEventListener('resize', resizeCanvas);
  }, []);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      drawOverlays();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [overlaysActive]);

  // Initialize camera
  const initCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }

      setIsLoading(false);
      setPermissionDenied(false);
      displayInstruction(assemblySteps[0], 3000);

    } catch (err) {
      console.error('Camera error:', err);
      setIsLoading(false);
      setPermissionDenied(true);
    }
  };

  // Setup voice recognition
  const setupVoiceRecognition = () => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = (event: any) => {
          const last = event.results.length - 1;
          const command = event.results[last][0].transcript.toLowerCase().trim();
          console.log('Voice command:', command);
          handleVoiceCommand(command);
        };

        recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error);
          if (event.error === 'not-allowed') {
            displayInstruction('Microphone access denied', 2000);
          }
        };

        recognition.onend = () => {
          if (isListening && recognitionRef.current) {
            recognitionRef.current.start();
          }
        };

        recognitionRef.current = recognition;
      }
    }
  };

  // Handle voice commands
  const handleVoiceCommand = (command: string) => {
    updateStatus(`Heard: "${command}"`);

    if (command.includes('start')) {
      setCurrentStep(1);
      displayInstruction(assemblySteps[1], 4000);
      speak(assemblySteps[1]);
    } 
    else if (command.includes('next')) {
      const nextStep = Math.min(currentStep + 1, assemblySteps.length - 1);
      setCurrentStep(nextStep);
      displayInstruction(assemblySteps[nextStep], 4000);
      speak(assemblySteps[nextStep]);
    }
    else if (command.includes('repeat')) {
      displayInstruction(assemblySteps[currentStep], 4000);
      speak(assemblySteps[currentStep]);
    }
    else if (command.includes('help')) {
      displayInstruction("Say 'start', 'next', 'repeat', or 'done'", 3000);
      speak("Available commands: start, next, repeat, or done");
    }
    else if (command.includes('done') || command.includes('complete')) {
      setCurrentStep(assemblySteps.length - 1);
      displayInstruction(assemblySteps[assemblySteps.length - 1], 4000);
      speak(assemblySteps[assemblySteps.length - 1]);
    }
    else if (command.includes('manual') || command.includes('upload')) {
      setShowManualPanel(true);
      speak("Manual upload panel opened");
    }
    else {
      displayInstruction(`Command not recognized: "${command}"`, 2000);
    }
  };

  // Text-to-speech
  const speak = (text: string) => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      speechSynthesis.speak(utterance);
    }
  };

  // Toggle voice listening
  const toggleVoice = () => {
    if (!recognitionRef.current) {
      displayInstruction('Voice recognition not supported', 2000);
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      updateStatus('Ready');
    } else {
      recognitionRef.current.start();
      setIsListening(true);
      updateStatus('Listening...');
    }
  };

  // Handle PDF upload
  const handlePDFUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || file.type !== 'application/pdf') {
      displayInstruction('Please upload a PDF file', 2000);
      return;
    }

    setIsProcessingManual(true);
    updateStatus('Processing manual...');

    try {
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL + "/upload-manual";
      if (!backendUrl) {
        throw new Error('Backend URL not configured');
      }

      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(backendUrl, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        setManualUploaded(true);
        setManualContent(data.content || 'Manual uploaded successfully');
        setDetectionMode('manual');
        displayInstruction('Manual processed! Now detecting based on manual', 3000);
        speak('Manual uploaded successfully. I will now guide you based on the manual instructions.');
        setShowManualPanel(false);
      } else {
        throw new Error(data.error || 'Failed to process manual');
      }

    } catch (error) {
      console.error('Manual upload error:', error);
      displayInstruction('Failed to upload manual', 2000);
    } finally {
      setIsProcessingManual(false);
      updateStatus('Ready');
    }
  };

  // Send frame to backend (with manual context if available)
  const sendFrameToBackend = async () => {
    if (!videoRef.current) return;

    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(videoRef.current, 0, 0);

    const imageData = canvas.toDataURL("image/jpeg");

    updateStatus('Analyzing...');

    try {
      const endpoint = detectionMode === 'manual' ? '/detect-with-manual' : '/detect';
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      const data = await response.json();
      
      if (data.detections) {
        setDetections(data.detections);
        
        if (data.instruction) {
          displayInstruction(data.instruction, 5000);
          speak(data.instruction);
        } else if (data.detections.length > 0) {
          const labels = data.detections.map((d: any) => d.label).join(', ');
          displayInstruction(`Found: ${labels}`, 3000);
          speak(`I found ${labels}`);
        }
      }

      updateStatus('Ready');

    } catch (error) {
      console.error('Detection error:', error);
      displayInstruction('Detection failed. Check backend connection.', 2000);
      updateStatus('Ready');
    }
  };

  // Toggle overlays
  const toggleOverlays = () => {
    setOverlaysActive(!overlaysActive);
    if (!overlaysActive) {
      displayInstruction("Test mode: Showing sample overlays", 2000);
    }
  };

  // Draw overlays on canvas
  const drawOverlays = () => {
    if (!canvasRef.current || detections.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.width;
    const height = canvas.height;

    detections.forEach((det) => {
      // Scale from backend resolution (assuming 1280x720)
      const x = (det.x / 1280) * width;
      const y = (det.y / 720) * height;
      const w = (det.width / 1280) * width;
      const h = (det.height / 720) * height;

      // Draw bounding box
      ctx.strokeStyle = det.color || "#22c55e";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);

      // Draw label background
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      const textMetrics = ctx.measureText(det.label);
      ctx.fillRect(x, y - 30, textMetrics.width + 20, 30);

      // Draw label text
      ctx.fillStyle = det.color || "#22c55e";
      ctx.font = "bold 18px Arial";
      ctx.fillText(det.label, x + 10, y - 8);

      // Draw pulsing circle for important items
      if (det.highlight) {
        const centerX = x + w / 2;
        const centerY = y + h / 2;
        const time = Date.now() / 1000;
        const radius = 30 + Math.sin(time * 2) * 10;

        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.strokeStyle = det.color || "#06b6d4";
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.8 - Math.abs(Math.sin(time * 2)) * 0.4;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
  };

  // Display instruction
  const displayInstruction = (text: string, duration: number) => {
    setInstruction(text);
    setShowInstruction(true);
    setTimeout(() => setShowInstruction(false), duration);
  };

  // Update status
  const updateStatus = (text: string) => {
    setStatusText(text);
    setTimeout(() => setStatusText('Ready'), 3000);
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black">
      {/* Loading Screen */}
      {isLoading && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black">
          <div className="w-20 h-20 bg-gradient-to-br from-cyan-400 to-purple-500 rounded-2xl flex items-center justify-center text-4xl font-black mb-5 animate-bounce">
            S
          </div>
          <div className="text-lg font-semibold text-white/80">Initializing camera...</div>
        </div>
      )}

      {/* Permission Denied Screen */}
      {permissionDenied && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black p-10 text-center">
          <div className="text-6xl mb-5">📷</div>
          <h1 className="text-2xl font-bold mb-3">Camera Access Required</h1>
          <p className="text-base text-white/70 leading-relaxed mb-8">
            AI needs camera access to guide you through assembly tasks. 
            Please allow camera permissions and try again.
          </p>
          <button 
            onClick={initCamera}
            className="px-10 py-4 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl font-bold text-lg hover:scale-105 transition-transform"
          >
            Try Again
          </button>
        </div>
      )}

      {/* Manual Upload Panel */}
      {showManualPanel && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/80 backdrop-blur-sm">
          <div className="bg-gradient-to-br from-zinc-900 to-black border-2 border-white/20 rounded-3xl p-8 max-w-md w-full mx-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold">Upload Assembly Manual</h2>
              <button 
                onClick={() => setShowManualPanel(false)}
                className="text-3xl text-white/60 hover:text-white"
              >
                ×
              </button>
            </div>

            <p className="text-white/70 mb-6 leading-relaxed">
              Upload your assembly manual PDF. The AI will analyze it and provide context-aware guidance based on the instructions.
            </p>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handlePDFUpload}
              className="hidden"
            />

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessingManual}
              className="w-full py-4 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl font-bold text-lg hover:scale-105 transition-transform disabled:opacity-50 disabled:cursor-not-allowed mb-4"
            >
              {isProcessingManual ? '⏳ Processing...' : '📄 Choose PDF Manual'}
            </button>

            {manualUploaded && (
              <div className="bg-green-500/20 border border-green-500/50 rounded-xl p-4">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">✅</span>
                  <div>
                    <div className="font-bold text-green-400">Manual Uploaded</div>
                    <div className="text-sm text-white/70">AI guidance is now manual-aware</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Video Container */}
      <div className="absolute inset-0">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
      </div>

      {/* Canvas for Overlays */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
      />

      {/* Top Bar */}
      <div className="absolute top-0 left-0 right-0 p-5 bg-gradient-to-b from-black/80 to-transparent z-10">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full">
              <div className={`w-2 h-2 rounded-full ${isListening ? 'bg-red-500' : 'bg-green-400'} animate-pulse`} />
              <span className="text-sm font-semibold">{statusText}</span>
            </div>
            {manualUploaded && (
              <div className="bg-purple-500/30 backdrop-blur-md px-3 py-1.5 rounded-full border border-purple-400/50">
                <span className="text-xs font-semibold">📖 Manual Mode</span>
              </div>
            )}
          </div>
          <button 
            onClick={() => setShowManualPanel(true)}
            className="w-10 h-10 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center text-xl hover:bg-white/20 transition-colors"
          >
            📄
          </button>
        </div>
      </div>

      {/* Detection Panel */}
      <div className="absolute top-20 left-5 bg-black/80 backdrop-blur-xl p-4 rounded-2xl border border-white/10 max-w-[200px] z-10">
        <div className="text-xs text-cyan-400 font-bold mb-2 uppercase tracking-wide">Detected</div>
        {detections.length === 0 ? (
          <div className="text-sm text-white/50 italic">No objects yet</div>
        ) : (
          detections.map((det, i) => (
            <div key={i} className="flex justify-between text-sm text-white my-1.5">
              <span>{det.label}</span>
              {det.confidence && (
                <span className="text-green-400 font-semibold">{Math.round(det.confidence * 100)}%</span>
              )}
            </div>
          ))
        )}
      </div>

      {/* Instruction Box */}
      <div 
        className={`absolute top-1/2 left-5 right-5 -translate-y-1/2 bg-gradient-to-r from-cyan-500/95 to-purple-500/95 backdrop-blur-xl p-6 rounded-2xl border-2 border-white/30 text-center z-10 transition-opacity duration-300 ${showInstruction ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
      >
        <p className="text-lg font-bold text-white leading-relaxed">
          &ldquo;{instruction}&rdquo;
        </p>
      </div>

      {/* Test Indicators */}
      {overlaysActive && (
        <div className="absolute bottom-48 left-5 right-5 flex flex-wrap gap-2 z-5">
          {testOverlays.map((overlay, i) => (
            <div key={i} className="bg-cyan-400/30 border border-cyan-400/60 px-3 py-1.5 rounded-full text-xs font-semibold">
              {overlay.type}: {overlay.label}
            </div>
          ))}
        </div>
      )}

      {/* Bottom Controls */}
      <div className="absolute bottom-0 left-0 right-0 p-8 pb-10 bg-gradient-to-t from-black/90 to-transparent z-10">
        {/* Control Buttons */}
        <div className="flex justify-center gap-4 mb-5">
          <button
            onClick={toggleVoice}
            className={`flex-1 max-w-[120px] p-4 rounded-2xl border-2 backdrop-blur-md transition-all ${
              isListening 
                ? 'bg-red-500/50 border-red-500/80' 
                : 'bg-white/10 border-white/20 active:scale-95'
            }`}
          >
            <div className="text-2xl mb-1.5">🎤</div>
            <div className="text-sm font-semibold">Voice</div>
          </button>

          <button
            onClick={sendFrameToBackend}
            className="flex-1 max-w-[120px] p-4 bg-white/10 backdrop-blur-md border-2 border-white/20 rounded-2xl active:scale-95 transition-all"
          >
            <div className="text-2xl mb-1.5">📸</div>
            <div className="text-sm font-semibold">Scan</div>
          </button>

          <button
            onClick={toggleOverlays}
            className="flex-1 max-w-[120px] p-4 bg-white/10 backdrop-blur-md border-2 border-white/20 rounded-2xl active:scale-95 transition-all"
          >
            <div className="text-2xl mb-1.5">🎯</div>
            <div className="text-sm font-semibold">Test</div>
          </button>
        </div>

        {/* Voice Commands List */}
        <div className="bg-white/5 backdrop-blur-md rounded-xl p-3 text-xs text-white/70">
          <div className="font-bold mb-2 text-purple-400">Voice Commands:</div>
          <div className="space-y-1">
            <div className="pl-3 relative before:content-['•'] before:absolute before:left-0 before:text-cyan-400">
              &quot;start&quot; - Begin guidance
            </div>
            <div className="pl-3 relative before:content-['•'] before:absolute before:left-0 before:text-cyan-400">
              &quot;next&quot; - Next step
            </div>
            <div className="pl-3 relative before:content-['•'] before:absolute before:left-0 before:text-cyan-400">
              &quot;manual&quot; - Upload manual
            </div>
            <div className="pl-3 relative before:content-['•'] before:absolute before:left-0 before:text-cyan-400">
              &quot;repeat&quot; - Repeat instruction
            </div>
            <div className="pl-3 relative before:content-['•'] before:absolute before:left-0 before:text-cyan-400">
              &quot;done&quot; - Mark complete
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}