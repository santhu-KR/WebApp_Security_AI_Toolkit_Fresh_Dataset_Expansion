import { GoogleGenAI, Type } from "@google/genai";
import type { Finding, ProjectFile } from "../types";

function getClient() {
 const key = (process.env.GEMINI_API_KEY || "").trim();
 if (!key || key.includes("your_google")) throw new Error("Gemini API key is missing. Set GEMINI_API_KEY in .env and restart the dev server.");
 return new GoogleGenAI({ apiKey: key });
}
const verificationSchema = {
 type: Type.ARRAY,
 items: { type: Type.OBJECT, properties: {
  id:{type:Type.STRING}, isTruePositive:{type:Type.BOOLEAN}, cweId:{type:Type.STRING},
  owaspCategory:{type:Type.STRING}, severity:{type:Type.STRING,enum:["Critical","High","Normal","Low"]},
  explanation:{type:Type.STRING}, fixedLineSnippet:{type:Type.STRING}
 }, required:["id","isTruePositive","cweId","owaspCategory","severity","explanation","fixedLineSnippet"] }
};

export async function verifyFindings(findings: Finding[], files: ProjectFile[] = []) {
 if (!findings.length) return [];
 const ai = getClient();
 const candidates = findings.map(({id,filePath,lineNumber,codeSnippet,mlConfidence,title})=>({
  id,filePath,lineNumber,codeSnippet,mlConfidence,title
 }));
 const relevantPaths = new Set(findings.map(f=>f.filePath));
 const fileContexts = files.filter(file=>relevantPaths.has(file.path)).map(file=>({
  filePath:file.path,completeSource:file.content.slice(0,30000),
  truncated:file.content.length>30000
 }));
 const response = await ai.models.generateContent({
  model:"gemini-2.5-flash",
  contents:`You are performing defensive source-code vulnerability triage. Review each candidate using the matching complete source-file context below, not just the matched line. Do not create exploit payloads or offensive instructions. Return exactly one result per candidate and preserve its id. Assess whether the code is genuinely vulnerable; trace relevant input sources, sanitization/validation, authorization checks, framework protections, and the sensitive sink where visible. Do not mark a finding confirmed merely because a suspicious keyword exists. If evidence is incomplete or the source file is marked truncated, set isTruePositive false and clearly say "Insufficient evidence" in the explanation. Identify CWE and OWASP Top 10 2021 category only when supported. Assign severity based on demonstrated impact and context. Provide a safe replacement for the exact flagged line when possible; otherwise return an empty fixedLineSnippet. Candidates:\n${JSON.stringify(candidates)}\n\nSource file contexts:\n${JSON.stringify(fileContexts)}`,
  config:{responseMimeType:"application/json",responseSchema:verificationSchema}
 });
 const raw = response.text || "[]";
 return JSON.parse(raw) as Array<Pick<Finding,"id"|"isTruePositive"|"cweId"|"owaspCategory"|"severity"|"explanation"|"fixedLineSnippet">>;
}

export async function chatWithGemini(file: ProjectFile, findings: Finding[], history: Array<{role:"user"|"model";text:string}>, prompt: string) {
 const ai = getClient();
 const fileFindings = findings
  .filter(f=>f.filePath===file.path)
  .map(f=>({lineNumber:f.lineNumber,title:f.title,severity:f.severity,cweId:f.cweId,explanation:f.explanation||f.predictedCategory,codeSnippet:f.codeSnippet}));

 const systemInstruction = `You are Gemini, acting as a senior defensive application security engineer.
The user is asking you to review or improve ONE selected source file in an authorized project.
SECURITY RULES:
- Only provide defensive code review, secure implementation, and remediation.
- Do not provide exploit payloads, offensive attack scripts, credential theft, or destructive actions.
FULL-FILE OUTPUT CONTRACT:
- Your response MUST contain the complete replacement source code for the selected file in exactly ONE fenced code block.
- Reproduce every required import, declaration, function, class, route, component, and unrelated existing feature from the supplied file.
- Do not return a diff, patch fragment, pseudocode, outline, or only the changed functions.
- Never use ellipses, "rest of code", omitted sections, TODO placeholders, or invented missing modules.
- Preserve the original language, framework, public interfaces, and existing behavior unless the user explicitly requests a change.
- Fix the listed security issues using safe, idiomatic patterns. If a finding is uncertain, explain the uncertainty and avoid breaking working behavior.
- After the single complete code block, add a short section titled "Changes made" and briefly explain the security fixes.
- If the file is too large to reproduce completely or the source context is incomplete, do NOT pretend the output is complete. State that clearly and ask the user to split the file or provide the missing context.`;

 const contents = [
  ...history.slice(-8).map(m=>({role:m.role,parts:[{text:m.text}]})),
  {role:"user" as const,parts:[{text:`TARGET FILE: ${file.path}
The following is the COMPLETE current content of the selected file. Treat it as the source of truth.

<complete_file>
${file.content}
</complete_file>

STATIC SCANNER FINDINGS FOR THIS FILE:
${JSON.stringify(fileFindings,null,2)}

USER REQUEST:
${prompt}

Return the COMPLETE replacement file, not a partial patch.`}]}
 ];

 const response = await ai.models.generateContent({
  model:"gemini-2.5-pro",
  contents,
  config:{
   systemInstruction,
   temperature:0.2,
   maxOutputTokens:32768
  }
 });
 return response.text || "Gemini returned an empty response. Please retry with a smaller file.";
}
