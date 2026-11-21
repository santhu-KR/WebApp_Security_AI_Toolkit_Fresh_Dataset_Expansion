import type { Finding, ProjectFile } from "../types";

const API = "http://127.0.0.1:5001/api";

async function postAI(path: string, token: string, body: unknown): Promise<string> {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `AI service returned HTTP ${response.status}`);
  return String(data.answer || "");
}

export async function explainFinding(finding: Finding, files: ProjectFile[], token: string): Promise<string> {
  const file = files.find((f) => f.path === finding.filePath);
  const context = file?.content || finding.codeSnippet || "";
  return postAI("/ai/explain", token, {
    finding: {
      title: finding.title,
      severity: finding.severity,
      category: finding.predictedCategory,
      ruleId: finding.ruleId,
      filePath: finding.filePath,
      lineNumber: finding.lineNumber,
      codeSnippet: finding.codeSnippet,
      mlPrediction: (finding as any).mlPrediction,
      mlConfidence: (finding as any).mlModelConfidence ?? finding.mlConfidence,
      cweId: finding.cweId,
      owaspCategory: finding.owaspCategory,
      previousVerification: finding.isTruePositive,
      previousExplanation: finding.explanation,
    },
    codeContext: context,
  });
}

export async function verifyFindings(findings: Finding[], files: ProjectFile[], token: string) {
  const verified: Array<Pick<Finding, "id" | "isTruePositive" | "cweId" | "owaspCategory" | "severity" | "explanation" | "fixedLineSnippet">> = [];
  for (const finding of findings) {
    try {
      const explanation = await explainFinding(finding, files, token);
      verified.push({
        id: finding.id,
        isTruePositive: !/false positive|not vulnerable|no vulnerability|insufficient evidence/i.test(explanation),
        cweId: finding.cweId,
        owaspCategory: finding.owaspCategory,
        severity: finding.severity,
        explanation,
        fixedLineSnippet: "",
      });
    } catch (error) {
      throw new Error(`AI analysis failed for ${finding.title}: ${String(error)}`);
    }
  }
  return verified;
}

export async function chatWithGemini(
  file: ProjectFile,
  findings: Finding[],
  history: Array<{ role: "user" | "model"; text: string }>,
  prompt: string,
  token: string
) {
  const fileFindings = findings
    .filter((f) => f.filePath === file.path)
    .map((f) => ({
      lineNumber: f.lineNumber,
      title: f.title,
      severity: f.severity,
      cweId: f.cweId,
      explanation: f.explanation || f.predictedCategory,
      codeSnippet: f.codeSnippet,
      mlPrediction: (f as any).mlPrediction,
      mlConfidence: (f as any).mlModelConfidence ?? f.mlConfidence,
    }));

  return postAI("/ai/chat", token, {
    filePath: file.path,
    fileContent: file.content,
    findings: fileFindings,
    history: history.slice(-8),
    prompt,
  });
}
