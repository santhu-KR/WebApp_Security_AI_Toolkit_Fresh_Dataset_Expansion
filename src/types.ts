export type Severity = "Critical" | "High" | "Normal" | "Low";
export type DetectionSource = "ML Feature Model" | "Static Rule" | "Hybrid (ML + Rule)";
export interface ProjectFile { path: string; content: string; originalContent: string; }
export interface Finding {
  id: string; fileName: string; filePath: string; lineNumber: number; codeSnippet: string;
  ruleId: string; title: string; predictedCategory: string; mlConfidence: number;
  detectionSource: DetectionSource; severity: Severity; isTruePositive?: boolean;
  cweId?: string; owaspCategory?: string; explanation?: string; fixedLineSnippet?: string;
}
export interface HeaderCheck { name: string; present: boolean; status: string; severity: Severity; detail: string; }
export interface ChatMessage { role: "user" | "model"; text: string; code?: string; }
export interface Snapshot { id: string; filePath: string; content: string; createdAt: string; label: string; }