import { BlogHeader } from "@/components/BlogHeader";
import { BlogContent } from "@/components/BlogContent";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <BlogHeader />
      <BlogContent />
    </div>
  );
};

export default Index;
