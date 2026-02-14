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

  return (
    <div className="relative w-full h-screen bg-black">
      {isLoading && <div className="absolute inset-0 flex items-center justify-center text-white">Initializing camera...</div>}
      {permissionDenied && <div className="absolute inset-0 flex items-center justify-center text-white">Camera access denied</div>}
      <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
    </div>
  );
}
