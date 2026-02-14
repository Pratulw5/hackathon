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

return (
  <>
    <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
    <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
  </>
);

}
