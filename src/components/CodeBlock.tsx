import { Card } from "@/components/ui/card";

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
}

export const CodeBlock = ({ code, language = "", title }: CodeBlockProps) => {
  return (
    <Card className="floating-card p-0 overflow-hidden">
      {title && (
        <div className="px-4 py-2 bg-muted/50 border-b border-border">
          <span className="text-sm font-mono text-muted-foreground">{title}</span>
        </div>
      )}
      <div className="p-4">
        <pre className="overflow-x-auto">
          <code className="text-sm font-mono text-code-foreground">
            {code}
          </code>
        </pre>
      </div>
    </Card>
  );
};