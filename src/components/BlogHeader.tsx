export const BlogHeader = () => {
  return (
    <div className="relative overflow-hidden">
      {/* Neural network background effect */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-primary rounded-full animate-pulse" />
        <div className="absolute top-1/3 right-1/3 w-1 h-1 bg-accent rounded-full animate-pulse delay-300" />
        <div className="absolute bottom-1/4 left-1/3 w-1.5 h-1.5 bg-primary-glow rounded-full animate-pulse delay-700" />
        <div className="absolute top-1/2 right-1/4 w-1 h-1 bg-accent rounded-full animate-pulse delay-1000" />
      </div>
      
      <div className="relative z-10 text-center py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-6xl font-bold mb-6 gradient-text leading-tight">
            GPU-Accelerated AI Inference
          </h1>
          <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
            Building Production-Ready Deep Learning Pipelines with NVIDIA Triton, 
            Docker Offload, and Agentic Architecture Patterns
          </p>
          
          <div className="flex items-center justify-center gap-8 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-primary rounded-full" />
              <span>Docker Offload</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-accent rounded-full" />
              <span>NVIDIA Triton</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-primary-glow rounded-full" />
              <span>Agentic AI</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};