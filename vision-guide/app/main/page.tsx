'use client';

import { useState, useEffect, useRef } from 'react';

export default function CameraApp() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [permissionDenied, setPermissionDenied] = useState(false);

  useEffect(() => {
    initCamera();
  }, []);

  const initCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false
      });
      if (videoRef.current) videoRef.current.srcObject = stream;
      setIsLoading(false);
      setPermissionDenied(false);
    } catch (err) {
      console.error('Camera error:', err);
      setIsLoading(false);
      setPermissionDenied(true);
    }
  };

  const canvasRef = useRef<HTMLCanvasElement>(null);
const [detections, setDetections] = useState<any[]>([]);

useEffect(() => {
  const animate = () => {
    drawOverlays();
    requestAnimationFrame(animate);
  };
  animate();
}, [detections]);

const drawOverlays = () => {
  if (!canvasRef.current || detections.length === 0) return;
  const ctx = canvasRef.current.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

  detections.forEach(det => {
    const x = det.x; const y = det.y; const w = det.width; const h = det.height;
    ctx.strokeStyle = det.color || "#22c55e";
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "white";
    ctx.fillText(det.label, x + 5, y - 5);
  });
};
const [isListening, setIsListening] = useState(false);
const recognitionRef = useRef<any>(null);
const [currentStep, setCurrentStep] = useState(0);
const [instruction, setInstruction] = useState('');
const [showInstruction, setShowInstruction] = useState(false);

const assemblySteps = [
  "Point your camera at the parts to begin",
  "Insert the screw into the marked hole",
  "Attach the left panel to the base",
  "Secure with the remaining screws",
  "Assembly complete! Great job!"
];

const displayInstruction = (text: string, duration: number) => {
  setInstruction(text);
  setShowInstruction(true);
  setTimeout(() => setShowInstruction(false), duration);
};

const handleVoiceCommand = (command: string) => {
  if (command.includes('start')) {
    setCurrentStep(1);
    displayInstruction(assemblySteps[1], 4000);
  } else if (command.includes('next')) {
    const nextStep = Math.min(currentStep + 1, assemblySteps.length - 1);
    setCurrentStep(nextStep);
    displayInstruction(assemblySteps[nextStep], 4000);
  }
};

useEffect(() => {
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
        handleVoiceCommand(command);
      };

      recognitionRef.current = recognition;
    }
  }
}, []);

const toggleVoice = () => {
  if (!recognitionRef.current) return;
  if (isListening) {
    recognitionRef.current.stop();
    setIsListening(false);
  } else {
    recognitionRef.current.start();
    setIsListening(true);
  }
};
return (
  <>
    <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
    <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
  </>
);

}
