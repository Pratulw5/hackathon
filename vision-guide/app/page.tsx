'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false);
  const [activeFeature, setActiveFeature] = useState(0);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  const features = [
    {
      title: 'See What You Need',
      description: 'AI-powered computer vision identifies every screw, panel, and tool in real-time',
      icon: '👁️',
      color: 'from-cyan-400 to-blue-500'
    },
    {
      title: 'Hands-Free Guidance',
      description: 'Voice commands keep your hands on the task while AI guides you',
      icon: '🎤',
      color: 'from-purple-400 to-pink-500'
    },
    {
      title: 'Visual Overlays',
      description: 'Pulsing markers and arrows show exactly where to place, insert, or turn',
      icon: '🎯',
      color: 'from-orange-400 to-red-500'
    },
    {
      title: 'Smart Assistance',
      description: 'Multi-modal AI understands context and adapts to your progress',
      icon: '🧠',
      color: 'from-green-400 to-emerald-500'
    }
  ];

  const stats = [
    { value: '95%', label: 'Faster Assembly' },
    { value: '0', label: 'Confusing Manuals' },
    { value: '∞', label: 'Patience' }
  ];

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      {/* Animated background */}
      <div className="fixed inset-0 opacity-30">
        <div className="absolute top-20 left-10 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob" />
        <div className="absolute top-40 right-10 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000" />
        <div className="absolute bottom-20 left-1/2 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-4000" />
      </div>

      {/* Grain texture overlay */}
      <div className="fixed inset-0 opacity-[0.03] pointer-events-none bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48ZmlsdGVyIGlkPSJhIiB4PSIwIiB5PSIwIj48ZmVUdXJidWxlbmNlIGJhc2VGcmVxdWVuY3k9Ii43NSIgc3RpdGNoVGlsZXM9InN0aXRjaCIgdHlwZT0iZnJhY3RhbE5vaXNlIi8+PGZlQ29sb3JNYXRyaXggdHlwZT0ic2F0dXJhdGUiIHZhbHVlcz0iMCIvPjwvZmlsdGVyPjxwYXRoIGQ9Ik0wIDBoMzAwdjMwMEgweiIgZmlsdGVyPSJ1cmwoI2EpIiBvcGFjaXR5PSIuMDUiLz48L3N2Zz4=')]" />

      <div className="relative z-10">
        {/* Header */}
        <header className="container mx-auto px-6 py-8">
          <nav className={`flex items-center justify-between transition-all duration-1000 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'}`}>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-purple-500 rounded-xl flex items-center justify-center font-black text-xl">
                AI
              </div>
              <span className="text-2xl font-black tracking-tight">AI Vision-Guide</span>
            </div>
            <div className="hidden md:flex items-center gap-8 text-sm font-medium text-zinc-400">
              <a href="#features" className="hover:text-white transition-colors">Features</a>
              <a href="#demo" className="hover:text-white transition-colors">Demo</a>
              <a href="#pricing" className="hover:text-white transition-colors">Pricing</a>
              <button className="px-6 py-2.5 bg-white text-black rounded-full hover:bg-zinc-200 transition-all font-semibold">
                Get Started
              </button>
            </div>
          </nav>
        </header>

        {/* Hero Section */}
        <section className="container mx-auto px-6 py-20 md:py-32">
          <div className="max-w-5xl mx-auto">
            <div className={`transition-all duration-1000 delay-200 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 backdrop-blur-sm border border-white/10 rounded-full mb-8">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-zinc-300">Now 100% Free & Open Source</span>
              </div>
              
              <h1 className="text-6xl md:text-8xl font-black tracking-tighter mb-6 bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent leading-[1.1]">
                Your AI Assembly
                <br />
                <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Assistant
                </span>
              </h1>
              
              <p className="text-xl md:text-2xl text-zinc-400 max-w-2xl mb-12 leading-relaxed font-light">
                Replace confusing instruction manuals with real-time AI guidance. 
                See exactly where every piece goes with visual overlays and voice commands.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 mb-16">
                <Link href="/main">
                  <button className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full font-bold text-lg hover:shadow-2xl hover:shadow-purple-500/50 transition-all hover:scale-105">
                    Get Started
                  </button>
                </Link>
                
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-8 max-w-2xl">
                {stats.map((stat, i) => (
                  <div key={i} className={`transition-all duration-1000 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`} style={{ transitionDelay: `${400 + i * 100}ms` }}>
                    <div className="text-4xl md:text-5xl font-black bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent mb-2">
                      {stat.value}
                    </div>
                    <div className="text-sm text-zinc-500 font-medium">
                      {stat.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Demo Preview */}
        <section className="container mx-auto px-6 py-20">
          <div className="max-w-5xl mx-auto">
            <div className="relative rounded-3xl overflow-hidden border border-white/10 bg-gradient-to-br from-zinc-900 to-black p-1">
              <div className="bg-black rounded-2xl overflow-hidden">
                {/* Phone mockup */}
                <div className="aspect-[9/16] max-w-sm mx-auto bg-gradient-to-br from-zinc-900 to-black relative">
                  {/* Camera view simulation */}
                  <div className="absolute inset-0 bg-zinc-800">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center space-y-4 p-8">
                        <div className="text-6xl">📱</div>
                        <p className="text-zinc-500 text-sm">Camera view with AR overlays</p>
                        <div className="flex justify-center gap-4 mt-8">
                          <div className="w-16 h-16 rounded-full border-4 border-cyan-400 animate-pulse" />
                          <div className="w-16 h-16 rounded-full border-4 border-purple-400 animate-pulse animation-delay-1000" />
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* UI elements */}
                  <div className="absolute top-0 left-0 right-0 p-6 bg-gradient-to-b from-black/80 to-transparent">
                    <div className="flex items-center justify-between">
                      <div className="text-xs font-medium text-white/80">✓ Ready</div>
                      <div className="flex gap-2">
                        <div className="w-8 h-8 rounded-full bg-white/10 backdrop-blur-sm" />
                        <div className="w-8 h-8 rounded-full bg-white/10 backdrop-blur-sm" />
                      </div>
                    </div>
                  </div>
                  
                  <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent">
                    <div className="bg-cyan-500/90 backdrop-blur-sm rounded-2xl p-4 mb-4">
                      <p className="text-white font-semibold text-sm">
                        "Insert the screw into the highlighted hole"
                      </p>
                    </div>
                    <div className="flex justify-center gap-4">
                      <button className="flex-1 py-3 bg-white/10 backdrop-blur-sm rounded-full font-medium text-sm">
                        🎤 Voice
                      </button>
                      <button className="flex-1 py-3 bg-white/10 backdrop-blur-sm rounded-full font-medium text-sm">
                        🔍 Scan
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="container mx-auto px-6 py-20">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-5xl md:text-6xl font-black mb-4 text-center">
              How It <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">Works</span>
            </h2>
            <p className="text-xl text-zinc-400 text-center mb-16 max-w-2xl mx-auto">
              Cutting-edge AI technology that makes assembly effortless
            </p>

            <div className="grid md:grid-cols-2 gap-6">
              {features.map((feature, i) => (
                <div
                  key={i}
                  className="group relative p-8 rounded-3xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 hover:border-white/20 transition-all duration-300 cursor-pointer overflow-hidden"
                  onMouseEnter={() => setActiveFeature(i)}
                >
                  <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-10 transition-opacity duration-500`} />
                  
                  <div className="relative z-10">
                    <div className="text-6xl mb-6">{feature.icon}</div>
                    <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                    <p className="text-zinc-400 leading-relaxed">{feature.description}</p>
                  </div>

                  <div className="absolute top-4 right-4 w-12 h-12 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-white/10 transition-colors">
                    <svg className="w-6 h-6 text-white/50 group-hover:text-white/80 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Tech Stack */}
        <section className="container mx-auto px-6 py-20">
          <div className="max-w-6xl mx-auto">
            <div className="bg-gradient-to-br from-white/5 to-white/[0.02] rounded-3xl p-12 border border-white/10">
              <h3 className="text-3xl font-bold mb-8 text-center">Built with Open Source</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                {[
                  { name: 'YOLOv8', desc: 'Object Detection' },
                  { name: 'Whisper', desc: 'Voice Recognition' },
                  { name: 'Gemini', desc: 'Multimodal AI' },
                  { name: 'React Native', desc: 'Mobile App' }
                ].map((tech, i) => (
                  <div key={i} className="text-center">
                    <div className="text-2xl font-bold mb-2">{tech.name}</div>
                    <div className="text-sm text-zinc-500">{tech.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="container mx-auto px-6 py-20 md:py-32">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-5xl md:text-7xl font-black mb-6 leading-tight">
              Ready to Build
              <br />
              <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Anything?
              </span>
            </h2>
            <p className="text-xl text-zinc-400 mb-12 max-w-2xl mx-auto">
              Deploy your own AI Vision-Guide in under 10 minutes. 
              100% free, forever.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="px-8 py-4 bg-white text-black rounded-full font-bold text-lg hover:bg-zinc-200 transition-all">
                View on GitHub
              </button>
              <button className="px-8 py-4 border-2 border-white/20 rounded-full font-bold text-lg hover:bg-white/5 transition-all">
                Read Documentation
              </button>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="container mx-auto px-6 py-12 border-t border-white/10">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-purple-500 rounded-lg flex items-center justify-center font-black">
                S
              </div>
              <span className="font-bold">AI Vision-Guide</span>
            </div>
            <div className="flex gap-6 text-sm text-zinc-500">
              <a href="#" className="hover:text-white transition-colors">Twitter</a>
              <a href="#" className="hover:text-white transition-colors">GitHub</a>
              <a href="#" className="hover:text-white transition-colors">Discord</a>
              <a href="#" className="hover:text-white transition-colors">Docs</a>
            </div>
          </div>
          <div className="text-center mt-8 text-sm text-zinc-600">
            © 2026  AI-Vision-Guide. Open source under MIT License.
          </div>
        </footer>
      </div>

      <style jsx>{`
        @keyframes blob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        .animation-delay-1000 {
          animation-delay: 1s;
        }
      `}</style>
    </div>
  );
}