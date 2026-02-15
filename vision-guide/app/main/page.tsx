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
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
const manualUploadedRef = useRef(false);
const stepsRef = useRef<any[]>([]);

  // State
  const [isLoading, setIsLoading] = useState(true);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [statusText, setStatusText] = useState('Listening...');
  const [instruction, setInstruction] = useState('');
  const [showInstruction, setShowInstruction] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [manualUploaded, setManualUploaded] = useState(false);
  const [showManualPanel, setShowManualPanel] = useState(true); // Show by default
  const [isProcessingManual, setIsProcessingManual] = useState(false);
  const [fps, setFps] = useState(0);
  
  // Tutorial mode state
  const [tutorialMode, setTutorialMode] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [steps, setSteps] = useState<any[]>([]);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [showStepCard, setShowStepCard] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  
  // Debug: Log when steps change
  useEffect(() => {
    console.log('Steps updated:', steps.length, steps);
  }, [steps]); const [isVoiceActive, setIsVoiceActive] = useState(false);

  useEffect(() => {
    drawOverlays();
  }, [detections]);

  // Initialize camera and start everything automatically
  useEffect(() => {
    initCamera();
    setupVoiceRecognition();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
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
  }, [detections]);

  // Auto-start detection when manual is uploaded
  useEffect(() => {
    if (manualUploaded) {
      startRealtimeDetection();
    }
  }, [manualUploaded]);

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
      
      // Auto-start voice recognition
      setTimeout(() => {
        if (recognitionRef.current) {
          try {
            recognitionRef.current.start();
            console.log('Voice recognition started');
            updateStatus('Listening...');
            speak('Voice assistant ready. Upload a manual to begin.');
          } catch (e) {
            console.log('Voice start error:', e);
          }
        }
      }, 1000);
    } catch (err) {
      console.error('Camera error:', err);
      setIsLoading(false);
      setPermissionDenied(true);
    }
  };
useEffect(() => {
  manualUploadedRef.current = manualUploaded;
}, [manualUploaded]);

useEffect(() => {
  stepsRef.current = steps;
}, [steps]);
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
          setIsVoiceActive(true); // Show active state
          handleVoiceCommand(command);
        };

        recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error);
        };

        recognition.onend = () => {
          // Always auto-restart recognition
          setTimeout(() => {
            try {
              if (recognitionRef.current && manualUploaded) {
                recognitionRef.current.start();
                setIsVoiceActive(true);
                console.log('Voice recognition restarted');
              }
            } catch (e) {
              console.log('Voice restart failed:', e);
              setIsVoiceActive(false);
            }
          }, 100);
        };

        recognitionRef.current = recognition;
        
        // Don't start until manual is uploaded
      }
    }
  };

  const handleVoiceCommand = async (command: string) => {
    updateStatus(`Processing: "${command}"`);

    // Check for tutorial start command
    if (command.includes('start') && (command.includes('tutorial') || command.includes('guide'))) {
      startTutorial();
      return;
    }

    // Route everything else to LLM assistant
    if (manualUploaded) {
      await askAssistant(command);
    }
  };

  const startTutorial = () => {
    console.log(`Starting tutorial - manualUploaded: ${manualUploaded}, steps.length: ${steps.length}`);
    console.log('Steps:', steps);
    
    if (!manualUploadedRef.current || stepsRef.current.length === 0)
 {
      speak("Please upload a manual first to start the tutorial.");
      displayInstruction("Upload a manual to begin", 3000);
      return;
    }

    setTutorialMode(true);
    setCurrentStepIndex(0);
    setCompletedSteps([]);
    setShowStepCard(true);
    
    const firstStep = stepsRef.current[0];

    speak(`Starting tutorial! Step 1: ${firstStep.instruction}`);
    displayInstruction(`Step 1: ${firstStep.instruction}`, 8000);
  };

  const completeCurrentStep = () => {
    const currentStep = currentStepIndex + 1;
    
    // Mark step as complete
    setCompletedSteps(prev => [...prev, currentStep]);
    
    // Celebrate completion
    speak(`Great job! Step ${currentStep} completed!`);
    displayInstruction(`✅ Step ${currentStep} Complete!`, 3000);
    
    // Move to next step after delay
    setTimeout(() => {
      if (currentStepIndex < steps.length - 1) {
        const nextIndex = currentStepIndex + 1;
        setCurrentStepIndex(nextIndex);
        const nextStep = steps[nextIndex];
        setShowStepCard(true);
        speak(`Next step: ${nextStep.instruction}`);
        displayInstruction(`Step ${nextIndex + 1}: ${nextStep.instruction}`, 8000);
      } else {
        // Tutorial complete
        speak("Congratulations! You've completed all steps. Assembly finished!");
        displayInstruction("🎉 Tutorial Complete!", 5000);
        setTutorialMode(false);
        setShowStepCard(false);
      }
    }, 2000);
  };

  const speak = async (text: string) => {
  try {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsVoiceActive(false);
    }

    const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (response.ok) {
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);

        if (recognitionRef.current && manualUploadedRef.current) {
          recognitionRef.current.start();
          setIsVoiceActive(true);
        }
      };

      audio.play();
    } else {
      fallbackSpeak(text);
    }
  } catch {
    fallbackSpeak(text);
  }
};


  const fallbackSpeak = (text: string) => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      speechSynthesis.speak(utterance);
    }
  };

  const startRealtimeDetection = () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    const detectionIntervalMs = 1000;
    let lastDetectionTime = Date.now();
    
    detectionIntervalRef.current = setInterval(async () => {
      const now = Date.now();
      const actualFps = 1000 / (now - lastDetectionTime);
      setFps(Math.round(actualFps * 10) / 10);
      lastDetectionTime = now;
      
      await sendFrameToBackend();
    }, detectionIntervalMs);

    updateStatus('Live & Listening');
  };

  const handlePDFUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || file.type !== 'application/pdf') {
      speak('Please upload a PDF file');
      return;
    }

    setIsProcessingManual(true);
    updateStatus('Processing manual...');

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL + "/upload-manual";
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(backendUrl, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        setManualUploaded(true);
        const uploadedSteps = data.steps || [];
        setSteps(uploadedSteps);
        setShowManualPanel(false);
        
        console.log(`Manual uploaded with ${uploadedSteps.length} steps:`, uploadedSteps);
        
        speak(`Manual uploaded successfully. Found ${data.parts_count} parts and ${data.steps_count} steps. I'm ready to help! Say "start tutorial" to begin step-by-step guidance.`);
        displayInstruction('Manual loaded! Say "Start tutorial" or ask me anything', 5000);
        
        // Start voice recognition now
        if (recognitionRef.current) {
          try {
            recognitionRef.current.start();
            setIsVoiceActive(true);
            console.log('Voice recognition started');
            updateStatus('Live & Listening');
          } catch (e) {
            console.error('Failed to start voice recognition:', e);
            setIsVoiceActive(false);
          }
        }
      } else {
        throw new Error(data.error || 'Failed to process manual');
      }
    } catch (error) {
      console.error('Manual upload error:', error);
      speak('Failed to upload manual');
    } finally {
      setIsProcessingManual(false);
      updateStatus('Live & Listening');
    }
  };

  const sendFrameToBackend = async () => {
    if (!videoRef.current || !videoRef.current.readyState || videoRef.current.readyState < 2) {
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(videoRef.current, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg", 0.7);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/detect-with-manual`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      const data = await response.json();
      
      if (data.detections) {
        setDetections(data.detections);
      }
    } catch (error) {
      console.error('Detection error:', error);
    }
  };

  const askAssistant = async (question: string) => {
    setIsProcessingVoice(true);

    try {
      // Get current frame
      let frameData = null;
      if (videoRef.current && videoRef.current.readyState >= 2) {
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0);
          frameData = canvas.toDataURL("image/jpeg", 0.6);
        }
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ask-assistant`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          current_step: currentStepIndex,
          detections: detections,
          frame: frameData,
          conversation_history: []
        })
      });

      const data = await response.json();
      
      if (data.answer) {
        speak(data.answer);
        
        if (data.step_instruction) {
          displayInstruction(data.step_instruction, 5000);
        }
        
        if (data.highlight_objects && data.highlight_objects.length > 0) {
          highlightObjectsInView(data.highlight_objects);
        }
      }
    } catch (error) {
      console.error('Assistant error:', error);
      speak('Sorry, I encountered an error. Please try again.');
    } finally {
      setIsProcessingVoice(false);
      updateStatus('Live & Listening');
    }
  };

  const highlightObjectsInView = (objectLabels: string[]) => {
    setDetections(prev => prev.map(det => ({
      ...det,
      highlight: objectLabels.some(label => 
        det.label.toLowerCase().includes(label.toLowerCase())
      ),
      color: objectLabels.some(label => 
        det.label.toLowerCase().includes(label.toLowerCase())
      ) ? '#f59e0b' : det.color
    })));
  };

  const drawOverlays = () => {
    if (!canvasRef.current || detections.length === 0) {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
      }
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.width;
    const height = canvas.height;

    detections.forEach((det) => {
      const x = (det.x / 1280) * width;
      const y = (det.y / 720) * height;
      const w = (det.width / 1280) * width;
      const h = (det.height / 720) * height;

      ctx.strokeStyle = det.color || "#22c55e";
      ctx.lineWidth = det.highlight ? 6 : 4;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      const textMetrics = ctx.measureText(det.label);
      ctx.fillRect(x, y - 30, textMetrics.width + 20, 30);

      ctx.fillStyle = det.color || "#22c55e";
      ctx.font = "bold 18px Arial";
      ctx.fillText(det.label, x + 10, y - 8);

      if (det.highlight) {
        const centerX = x + w / 2;
        const centerY = y + h / 2;
        const time = Date.now() / 1000;
        const radius = 40 + Math.sin(time * 3) * 15;

        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.strokeStyle = det.color || "#f59e0b";
        ctx.lineWidth = 4;
        ctx.globalAlpha = 0.9 - Math.abs(Math.sin(time * 3)) * 0.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
  };

  const displayInstruction = (text: string, duration: number) => {
    setInstruction(text);
    setShowInstruction(true);
    setTimeout(() => setShowInstruction(false), duration);
  };

  const updateStatus = (text: string) => {
    setStatusText(text);
  };

  const progressPercentage = steps.length > 0 ? (completedSteps.length / steps.length) * 100 : 0;

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
            AI needs camera access to guide you through assembly.
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
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/90 backdrop-blur-sm">
          <div className="bg-gradient-to-br from-zinc-900 to-black border-2 border-cyan-400/30 rounded-3xl p-8 max-w-md w-full mx-6">
            <div className="text-center mb-6">
              <div className="w-16 h-16 bg-gradient-to-br from-cyan-400 to-purple-500 rounded-2xl flex items-center justify-center text-3xl font-black mb-4 mx-auto">
                📖
              </div>
              <h2 className="text-2xl font-bold mb-2">Upload Your Manual</h2>
              <p className="text-white/60 text-sm">
                Upload your assembly manual to get started with AI-powered guidance
              </p>
            </div>

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
              className="w-full py-4 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl font-bold text-lg hover:scale-105 transition-transform disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessingManual ? '⏳ Processing...' : '📄 Choose PDF Manual'}
            </button>

            <div className="mt-6 text-xs text-white/50 text-center">
              <p>✓ Real-time object detection</p>
              <p>✓ Voice-activated assistant</p>
              <p>✓ Step-by-step tutorial mode</p>
            </div>
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

      {/* Top Status Bar */}
      <div className="absolute top-0 left-0 right-0 p-5 bg-gradient-to-b from-black/80 to-transparent z-10">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-sm font-semibold">{statusText}</span>
            </div>
            {fps > 0 && (
              <div className="bg-red-500/30 backdrop-blur-md px-3 py-1.5 rounded-full border border-red-400/50">
                <span className="text-xs font-semibold">🔴 LIVE • {fps} FPS</span>
              </div>
            )}
            {isProcessingVoice && (
              <div className="bg-blue-500/30 backdrop-blur-md px-3 py-1.5 rounded-full border border-blue-400/50">
                <span className="text-xs font-semibold">🤖 Thinking...</span>
              </div>
            )}
          </div>
          {manualUploaded && (
            <div className="bg-purple-500/30 backdrop-blur-md px-4 py-2 rounded-full border border-purple-400/50">
              <span className="text-xs font-semibold">📖 Manual Loaded</span>
            </div>
          )}
        </div>
      </div>

      {/* Progress Bar (Tutorial Mode) */}
      {tutorialMode && (
        <div className="absolute top-20 left-5 right-5 z-10">
          <div className="bg-black/80 backdrop-blur-xl rounded-2xl border border-white/10 p-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-bold text-cyan-400">Tutorial Progress</span>
              <span className="text-sm text-white/70">
                {completedSteps.length} / {steps.length} steps
              </span>
            </div>
            <div className="w-full h-3 bg-white/10 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-500 ease-out"
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Current Step Card (Tutorial Mode) */}
      {tutorialMode && showStepCard && currentStepIndex < steps.length && (
        <div className="absolute bottom-24 left-5 right-5 z-10">
          <div className="bg-gradient-to-br from-zinc-900/95 to-black/95 backdrop-blur-xl rounded-2xl border-2 border-cyan-400/30 p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <div className="text-cyan-400 text-xs font-bold uppercase tracking-wide mb-1">
                  Step {currentStepIndex + 1} of {steps.length}
                </div>
                <h3 className="text-lg font-bold text-white leading-relaxed">
                  {steps[currentStepIndex].instruction}
                </h3>
              </div>
            </div>
            
            <button
              onClick={completeCurrentStep}
              className="w-full py-3 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl font-bold text-white hover:scale-105 transition-transform active:scale-95"
            >
              ✓ Mark as Complete
            </button>
          </div>
        </div>
      )}

      {/* Instruction Overlay */}
      <div 
        className={`absolute top-1/2 left-5 right-5 -translate-y-1/2 bg-gradient-to-r from-cyan-500/95 to-purple-500/95 backdrop-blur-xl p-6 rounded-2xl border-2 border-white/30 text-center z-10 transition-opacity duration-300 ${showInstruction ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
      >
        <p className="text-lg font-bold text-white leading-relaxed">
          {instruction}
        </p>
      </div>

      {/* Detection Panel */}
      <div className="absolute top-40 left-5 bg-black/80 backdrop-blur-xl p-4 rounded-2xl border border-white/10 max-w-[200px] z-10">
        <div className="text-xs text-cyan-400 font-bold mb-2 uppercase tracking-wide">Detected</div>
        {detections.length === 0 ? (
          <div className="text-sm text-white/50 italic">Scanning...</div>
        ) : (
          detections.slice(0, 5).map((det, i) => (
            <div key={i} className={`flex justify-between text-sm my-1.5 ${det.highlight ? 'text-yellow-400 font-bold' : 'text-white'}`}>
              <span className="truncate mr-2">{det.label}</span>
              {det.confidence && (
                <span className="text-green-400 font-semibold whitespace-nowrap text-xs">
                  {Math.round(det.confidence * 100)}%
                </span>
              )}
            </div>
          ))
        )}
      </div>

      {/* Voice Indicator */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10">
        <div className={`backdrop-blur-xl rounded-full px-6 py-4 border-2 flex items-center gap-3 transition-all shadow-lg ${
          isVoiceActive 
            ? 'bg-green-500/20 border-green-400/60 shadow-green-400/30' 
            : 'bg-black/90 border-cyan-400/30'
        }`}>
          <div className="relative flex items-center justify-center">
            {isVoiceActive && (
              <>
                <div className="absolute w-6 h-6 rounded-full bg-green-400/30 animate-ping" />
                <div className="absolute w-5 h-5 rounded-full bg-green-400/50 animate-pulse" />
              </>
            )}
            <div className={`w-3 h-3 rounded-full z-10 ${isVoiceActive ? 'bg-green-400' : 'bg-gray-400'}`} />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-bold text-white">
              {isVoiceActive ? '🎤 Listening...' : '🎤 Voice Inactive'}
            </span>
            {isVoiceActive && (
              <span className="text-xs text-green-300">
                Speak to ask questions
              </span>
            )}
          </div>
          {!isVoiceActive && manualUploaded && (
            <button
              onClick={() => {
                if (recognitionRef.current) {
                  try {
                    recognitionRef.current.start();
                    setIsVoiceActive(true);
                    speak('Voice assistant activated');
                  } catch (e) {
                    console.log('Voice start error:', e);
                    setIsVoiceActive(false);
                  }
                }
              }}
              className="ml-2 px-4 py-1.5 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full text-xs font-bold hover:scale-105 transition-transform"
            >
              ACTIVATE
            </button>
          )}
        </div>
      </div>
    </div>
  );
}