import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ArchitectureDiagramProps {
  title: string;
  mermaidCode: string;
}

export const ArchitectureDiagram = ({ title, mermaidCode }: ArchitectureDiagramProps) => {
  return (
    <Card className="floating-card">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-primary">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="bg-card border border-border rounded-lg p-6">
          <div dangerouslySetInnerHTML={{ __html: `<lov-mermaid>${mermaidCode}</lov-mermaid>` }} />
        </div>
      </CardContent>
    </Card>
  );
};