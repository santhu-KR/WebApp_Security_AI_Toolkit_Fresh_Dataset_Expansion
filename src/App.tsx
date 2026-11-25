import React, { useEffect, useMemo, useRef, useState } from "react";

import JSZip from "jszip";

import { Activity, ShieldCheck, FolderOpen, Search, MessageSquare, LayoutDashboard, FileCode2, Upload, Play, Download, RefreshCw, ChevronRight, AlertTriangle, CheckCircle2, Copy, RotateCcw, Globe, Sparkles, Send, X, Code2, FileArchive, Lock, KeyRound, Github, LogOut } from "lucide-react";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

import type { ChatMessage, Finding, HeaderCheck, ProjectFile, Severity, Snapshot } from "./types";

import { scanProject, normalizePath } from "./engine/scanner";

import { verifyFindings, chatWithGemini, explainFinding } from "./engine/gemini";

import { inspectHeaders, inspectEndpoint } from "./engine/headers";



type Tab = "dashboard"|"scanner"|"findings"|"chat";

const sampleFiles: ProjectFile[] = [

 {path:"backend/auth.py",originalContent:`import hashlib\nfrom flask import request\n\nSECRET_KEY = "sk_live_4f9a8d7c6b5e4d3c2b1a"\nDEBUG = True\n\ndef login(username, password):\n    password_hash = hashlib.md5(password.encode()).hexdigest()\n    query = "SELECT * FROM users WHERE username = '" + username + "'"\n    return db.execute(query)\n`,content:""},

 {path:"backend/db_queries.py",originalContent:`import sqlite3\n\ndef get_user(conn, user_id):\n    query = f"SELECT * FROM users WHERE id = {user_id}"\n    return conn.execute(query).fetchone()\n`,content:""},

 {path:"frontend/ProfileCard.tsx",originalContent:`import React from "react";\n\nexport function ProfileCard({ bio }: { bio: string }) {\n  return <div className="profile" dangerouslySetInnerHTML={{ __html: bio }} />;\n}\n`,content:""},

 {path:"config/.env.example",originalContent:`APP_ENV=development\nAPI_KEY="AIzaSyD-EXAMPLE-1234567890abcdef"\nSESSION_SECRET="dev_session_secret_1234567890"\n`,content:""}

].map(f=>({...f,content:f.originalContent}));

const nav = [

 {id:"dashboard" as Tab,label:"Security Dashboard",icon:LayoutDashboard},

 {id:"scanner" as Tab,label:"Project Folder & URL Scanner",icon:FolderOpen},

 {id:"findings" as Tab,label:"Findings Explorer",icon:Search},

 {id:"chat" as Tab,label:"Gemini AI Security Chat",icon:MessageSquare}

];

const severityOrder: Severity[]=["Critical","High","Normal","Low"];

const sevColor: Record<Severity,string>={Critical:"#ef4444",High:"#f97316",Normal:"#f59e0b",Low:"#3b82f6"};

const fmt=(n:number)=>new Intl.NumberFormat().format(n);

const emptyNote="No scan data yet. Load the sample project or upload source files to begin.";



function App(){

 const [tab,setTab]=useState<Tab>("dashboard");

 const [files,setFiles]=useState<ProjectFile[]>([]);

 const [findings,setFindings]=useState<Finding[]>([]);

 const [patchedFiles,setPatchedFiles]=useState<Set<string>>(new Set());

 const [busy,setBusy]=useState(false);

 const [mlStatus,setMlStatus]=useState<{available:boolean;message:string;algorithm?:string;datasetRows?:number|null}>({available:false,message:"Checking local ML service…"});

 const [banner,setBanner]=useState("");

 const [query,setQuery]=useState("");

 const [severityFilter,setSeverityFilter]=useState("All");

 const [selected,setSelected]=useState<Finding|null>(null);

 const [rawHeaders,setRawHeaders]=useState("");

 const [headerChecks,setHeaderChecks]=useState<HeaderCheck[]>([]);

 const [endpoint,setEndpoint]=useState("");

 const [authorized,setAuthorized]=useState(false);

 const [activeFile,setActiveFile]=useState("");

 const [chatInput,setChatInput]=useState("");

 const [chatMessages,setChatMessages]=useState<ChatMessage[]>([]);

 const [snapshots,setSnapshots]=useState<Snapshot[]>([]);

 const [generatedCode,setGeneratedCode]=useState("");

 const [chatBusy,setChatBusy]=useState(false);
 const [explainBusy,setExplainBusy]=useState(false);

 const [authToken,setAuthToken]=useState(()=>sessionStorage.getItem("sentinel_token")||"");

 const [authEmail,setAuthEmail]=useState("");

 const [authPassword,setAuthPassword]=useState("");

 const [authMode,setAuthMode]=useState<"login"|"register">("login");

 const [authBusy,setAuthBusy]=useState(false);

 const [githubUrl,setGithubUrl]=useState("");

 const [githubBusy,setGithubBusy]=useState(false);

 const [signedInEmail,setSignedInEmail]=useState("");
 const [scanHistory,setScanHistory]=useState<any[]>(()=>{try{return JSON.parse(localStorage.getItem("sentinel_scan_history")||"[]");}catch{return []; }});
 const securityScore = useMemo(() => {
  const deductions: Record<string, number> = {
   Critical: 25,
   High: 15,
   Normal: 8,
   Low: 3,
  };
  const totalDeduction = findings.reduce(
   (sum, f) => sum + (deductions[f.severity] || 0),
   0
  );
  return Math.max(0, Math.min(100, 100 - totalDeduction));
 }, [findings]);
 const scoreLabel=securityScore>=90?"Excellent":securityScore>=75?"Good":securityScore>=50?"Needs Improvement":"At Risk";


 const API="http://127.0.0.1:5001";

 const folderRef=useRef<HTMLInputElement>(null);

 const zipRef=useRef<HTMLInputElement>(null);



 const counts=useMemo(()=>severityOrder.map(s=>({name:s,value:findings.filter(f=>f.severity===s).length,fill:sevColor[s]})),[findings]);

 const fileRisk=useMemo(()=>files.map(file=>({path:file.path,count:findings.filter(f=>f.filePath===file.path&&f.isTruePositive!==false).length,critical:findings.filter(f=>f.filePath===file.path&&["Critical","High"].includes(f.severity)&&f.isTruePositive!==false).length})).sort((a,b)=>b.count-a.count),[files,findings]);

 const visibleFindings=useMemo(()=>findings.filter(f=>(severityFilter==="All"||f.severity===severityFilter)&&`${f.filePath} ${f.title} ${f.predictedCategory}`.toLowerCase().includes(query.toLowerCase())).sort((a,b)=>severityOrder.indexOf(a.severity)-severityOrder.indexOf(b.severity)||a.filePath.localeCompare(b.filePath)||a.lineNumber-b.lineNumber),[findings,severityFilter,query]);

 const currentFile=files.find(f=>f.path===activeFile)||files[0];

 useEffect(()=>{let alive=true;fetch("http://127.0.0.1:5001/api/ml/status").then(r=>r.json()).then(data=>{if(alive)setMlStatus({available:!!data.available,message:data.message||"ML service status received",algorithm:data.algorithm,datasetRows:data.datasetRows});}).catch(()=>{if(alive)setMlStatus({available:false,message:"Start the Python ML service to enable trained-model scoring."});});return()=>{alive=false;};},[]);

 useEffect(()=>{if(!authToken)return;let alive=true;fetch(`${API}/api/auth/me`,{headers:{Authorization:`Bearer ${authToken}`}}).then(async r=>{if(!r.ok)throw new Error("Session expired");return r.json();}).then(data=>{if(alive)setSignedInEmail(data.user.email);}).catch(()=>{if(alive){sessionStorage.removeItem("sentinel_token");setAuthToken("");setSignedInEmail("");}});return()=>{alive=false;};},[authToken]);

 async function submitAuth(){

  setAuthBusy(true);

  try{

   const response=await fetch(`${API}/api/auth/${authMode}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email:authEmail,password:authPassword})});

   const data=await response.json();

   if(!response.ok)throw new Error(data.error||"Authentication failed.");

   sessionStorage.setItem("sentinel_token",data.token);setAuthToken(data.token);setSignedInEmail(data.user.email);setAuthPassword("");setBanner(`Signed in as ${data.user.email}.`);

  }catch(e){setBanner(String(e));}

  finally{setAuthBusy(false);}

 }

 function signOut(){

  sessionStorage.removeItem("sentinel_token");setAuthToken("");setSignedInEmail("");setFiles([]);setFindings([]);setChatMessages([]);setSnapshots([]);setBanner("You have signed out.");

 }

 async function importGithub(){

  if(!githubUrl.trim())return;

  setGithubBusy(true);

  try{

   const response=await fetch(`${API}/api/github/import`,{method:"POST",headers:{"Content-Type":"application/json","Authorization":`Bearer ${authToken}`},body:JSON.stringify({url:githubUrl})});

   const data=await response.json();

   if(!response.ok)throw new Error(data.error||`GitHub import failed (${response.status})`);

   const next:ProjectFile[]=(data.files||[]).map((f:any)=>({path:normalizePath(f.path),content:f.content,originalContent:f.content}));

   await loadFiles(next);setBanner(`Imported ${data.repository} (${data.defaultBranch}): ${next.length} supported files. ${data.note}`);

   setTab("dashboard");

  }catch(e){setBanner(`GitHub import failed: ${String(e)}`);}

  finally{setGithubBusy(false);}

 }

 const fileCounts=(path:string)=>findings.filter(f=>f.filePath===path&&f.isTruePositive!==false).length;




 function saveScanHistory(scanFiles:ProjectFile[],scanFindings:Finding[]){
  const deductions: Record<string, number> = { Critical: 25, High: 15, Normal: 8, Low: 3 };
  const totalDeduction = scanFindings.reduce(
    (sum, f) => sum + (deductions[f.severity] || 0),
    0
  );
  const score = Math.max(0, Math.min(100, 100 - totalDeduction));
  const entry={id:crypto.randomUUID(),date:new Date().toISOString(),project:scanFiles.length?scanFiles[0].path.split("/")[0]||"Uploaded project":"Project scan",files:scanFiles.length,total:scanFindings.length,critical:scanFindings.filter(f=>f.severity==="Critical").length,high:scanFindings.filter(f=>f.severity==="High").length,medium:scanFindings.filter(f=>f.severity==="Normal").length,low:scanFindings.filter(f=>f.severity==="Low").length,score};
  setScanHistory(old=>{const next=[entry,...old].slice(0,25);localStorage.setItem("sentinel_scan_history",JSON.stringify(next));return next;});
 }

 async function loadFiles(next:ProjectFile[]){

  const ignoredFolders = ["node_modules", ".git", "dist", "build"];
  const filtered = next.filter((f) => {
    const normalizedPath = f.path.replaceAll("\\", "/");
    const parts = normalizedPath.split("/");
    return !parts.some((part) => ignoredFolders.includes(part)) && !f.content.includes("\0");
  });

  setFiles(filtered.map(f=>({...f,originalContent:f.originalContent??f.content})));

  const results=scanProject(filtered);

  setFindings(results);setPatchedFiles(new Set());setSelected(results[0]||null);setActiveFile(filtered[0]?.path||"");saveScanHistory(filtered,results);

  setBanner(`Loaded ${filtered.length} source file(s). ${results.length} static candidate(s) identified. Scoring with trained ML model…`);

  if(results.length){

   try{

    const response=await fetch("http://127.0.0.1:5001/api/ml/predict",{method:"POST",headers:{"Content-Type":"application/json","Authorization":`Bearer ${authToken}`},body:JSON.stringify({items:results.map(f=>({id:f.id,code:f.codeSnippet}))})});

    if(!response.ok)throw new Error((await response.json()).error||`ML service returned ${response.status}`);

    const data=await response.json();const byId=new Map((data.predictions||[]).map((x:any)=>[x.id,x]));

    const scored=results.map(f=>{const prediction:any=byId.get(f.id);return prediction?{...f,mlConfidence:prediction.confidencePercent,mlModelConfidence:prediction.confidencePercent,mlPrediction:prediction.prediction,mlModelName:data.modelName}:f;});

    setFindings(scored);setSelected(scored[0]||null);

    setBanner(`Trained ML model scored ${scored.length} candidate(s). Probabilities are experimental and must be validated.`);

   }catch(error){

    setMlStatus({available:false,message:"Trained model API unavailable"});

    setBanner(`Static scan completed. Trained ML scoring unavailable: ${String(error)}. Start ml_backend/app.py to enable it.`);

   }

  }else{setBanner(`Loaded ${filtered.length} file(s). No static candidate findings found.`);}

 }

 function loadSample(){void loadFiles(sampleFiles);setTab("dashboard");}

 async function folderUpload(list:FileList|null){

  if(!list)return;setBusy(true);

  try {
    const next: ProjectFile[] = [];
    for (const f of Array.from(list)) {
      const path = normalizePath((f as any).webkitRelativePath || f.name);
      if (!/\.(py|js|ts|tsx|jsx|php|java|html|env|json|yml|yaml)$/i.test(path)) continue;
      const content = await f.text();
      if (!content.includes("\0")) next.push({ path, content, originalContent: content });
    }
    await loadFiles(next);
  }

  catch(e){setBanner(`Upload failed: ${String(e)}`)}finally{setBusy(false);}

 }

 async function zipUpload(file?:File){

  if(!file)return;setBusy(true);

  try {
    const zip = await JSZip.loadAsync(file);
    const next: ProjectFile[] = [];
    for (const [path, entry] of Object.entries(zip.files)) {
      if (entry.dir || /(^|\/)(node_modules|\.git|dist|build)(\/|$)/i.test(path) || !/\.(py|js|ts|tsx|jsx|php|java|html|env|json|yml|yaml)$/i.test(path)) continue;
      const content = await entry.async("string");
      if (!content.includes("\0")) next.push({ path: normalizePath(path), content, originalContent: content });
    }
    await loadFiles(next);
  }

  catch(e){setBanner(`ZIP could not be read: ${String(e)}`)}finally{setBusy(false);}

 }

 async function runVerification(){

  if(!findings.length){setBanner("There are no candidate findings to verify.");return;}

  setBusy(true);setBanner("Gemini is reviewing candidate findings…");

  try{const results=await verifyFindings(findings,files,authToken);const map=new Map(results.map(r=>[r.id,r]));setFindings(old=>old.map(f=>({...f,...(map.get(f.id)||{})})));setBanner(`Gemini verification complete. ${results.filter(r=>r.isTruePositive).length} true positive(s) reported.`);}

  catch(e){setBanner(String(e).includes("429")?"Gemini rate limit reached. Wait and retry.":`Gemini verification failed: ${String(e)}`)}

  finally{setBusy(false);}

 }


 async function explainSelectedFinding(){
  if(!selected)return;
  setExplainBusy(true);
  try{const explanation=await explainFinding(selected,files,authToken);setFindings(old=>old.map(f=>f.id===selected.id?{...f,explanation}:f));setSelected(old=>old?{...old,explanation}:old);setBanner("AI explanation generated using the finding and source context.");}
  catch(e){setBanner(`AI explanation failed: ${String(e)}`);}
  finally{setExplainBusy(false);}
 }

 async function inspectRaw(){const checks=inspectHeaders(rawHeaders);setHeaderChecks(checks);setBanner(`Inspected headers: ${checks.filter(c=>!c.present).length} missing or insecure control(s).`);}

 async function inspectUrl(){

  if(!authorized){setBanner("Confirm that you own or are authorized to inspect this endpoint.");return;}

  setBusy(true);try{const raw=await inspectEndpoint(endpoint);setRawHeaders(raw);setHeaderChecks(inspectHeaders(raw));setBanner("Passive CORS-enabled endpoint inspection complete.");}

  catch(e){setBanner(`Endpoint inspection failed (CORS/network restrictions may apply): ${String(e)}`)}finally{setBusy(false);}

 }

 async function sendChat(text=chatInput){

  if(!currentFile||!text.trim())return;setChatBusy(true);setChatMessages(old=>[...old,{role:"user",text}]);setChatInput("");

   try{const response=await chatWithGemini(currentFile,findings,chatMessages.filter(m=>m.role==="user"||m.role==="model").map(m=>({role:m.role,text:m.text})),text,authToken);const match=response.match(/```(?:[\w+-]+)?\s*\n([\s\S]*?)```/);setGeneratedCode(match?.[1]?.trimEnd()||"");setChatMessages(old=>[...old,{role:"model",text:response,code:match?.[1]?.trimEnd()}]);}

  catch(e){setChatMessages(old=>[...old,{role:"model",text:`Gemini request failed: ${String(e)}. Check the local API key, model access, network, and quota.`}]);}

  finally{setChatBusy(false);}

 }

 function applyCode(){

  if(!currentFile||!generatedCode)return;

  const snapshot:Snapshot={id:crypto.randomUUID(),filePath:currentFile.path,content:currentFile.content,createdAt:new Date().toISOString(),label:"Before Gemini patch"};

  setSnapshots(old=>[snapshot,...old]);setFiles(old=>old.map(f=>f.path===currentFile.path?{...f,content:generatedCode}:f));

  setPatchedFiles(old=>new Set(old).add(currentFile.path));

  const updated=files.map(f=>f.path===currentFile.path?{...f,content:generatedCode}:f);setFindings(scanProject(updated));setBanner(`Applied generated full-file code to ${currentFile.path}. A rollback snapshot was saved.`);

 }

 function rollback(s:Snapshot){

  setFiles(old=>old.map(f=>f.path===s.filePath?{...f,content:s.content}:f));setPatchedFiles(old=>{const n=new Set(old);n.delete(s.filePath);return n;});

  const updated=files.map(f=>f.path===s.filePath?{...f,content:s.content}:f);setFindings(scanProject(updated));setBanner(`Rolled back ${s.filePath} to ${new Date(s.createdAt).toLocaleString()}.`);

 }

 function downloadReport() {
  const body = [
    "# Web Application Security Assessment",
    "",
    `Generated: ${new Date().toLocaleString()}`,
    "",
    "## Summary",
    `- Files scanned: ${files.length}`,
    `- Candidate findings: ${findings.length}`,
    `- Patched files: ${patchedFiles.size}`,
    "",
    "## Findings",
    ...findings.map(
      (f) =>
        `### ${f.severity}: ${f.title}\n` +
        `- Location: ${f.filePath} : Line ${f.lineNumber}\n` +
        `- Confidence: ${f.mlConfidence}%\n` +
        `- Detection: ${f.detectionSource}\n` +
        `- CWE: ${f.cweId || "Not verified"}\n` +
        `- OWASP: ${f.owaspCategory || "Not verified"}\n` +
        `- True positive: ${f.isTruePositive === undefined ? "Not verified" : f.isTruePositive}\n` +
        `- Explanation: ${f.explanation || "Static candidate; Gemini verification not run."}\n` +
        `- Evidence:\n\n` +
        "```\n" +
        `${f.codeSnippet}\n` +
        "```\n"
    ),
    "",
    "## Passive Header Inspection",
    ...headerChecks.map(
      (c) => `- ${c.present ? "PASS" : "REVIEW"} — ${c.name}: ${c.status}`
    ),
  ].join("\n");

  downloadBlob(
    new Blob([body], { type: "text/markdown" }),
    "security-assessment.md",
    "text/markdown"
  );
 }

 async function downloadZip(){

  const zip=new JSZip();for(const f of files)zip.file(f.path,f.content);

  zip.file("SECURITY-ASSESSMENT.md",`# Security Assessment\n\nFiles: ${files.length}\nFindings: ${findings.length}\nPatched files: ${patchedFiles.size}\n`);

  const blob=await zip.generateAsync({type:"blob"});downloadBlob(blob,"patched-project.zip","application/zip");

 }

 function downloadBlob(data:BlobPart,name:string,type:string){const blob=data instanceof Blob?data:new Blob([data],{type});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);}

 async function copy(text:string){await navigator.clipboard.writeText(text);setBanner("Copied to clipboard.");}



 if(!authToken){

  return <div className="min-h-screen bg-slate-950 text-slate-200 flex items-center justify-center p-5">

   <div className="w-full max-w-md panel p-7">

    <div className="flex items-center gap-3 mb-6"><div className="rounded-xl bg-emerald-500/15 p-3 text-emerald-400"><ShieldCheck size={26}/></div><div><div className="font-bold text-white text-lg">SENTINEL AI</div><div className="text-[10px] tracking-[.2em] text-slate-500">APPLICATION DEFENSE WORKSPACE</div></div></div>

    <h1 className="text-xl font-semibold text-white">{authMode==="login"?"Sign in to your workspace":"Create your account"}</h1>

    <p className="text-xs text-slate-500 mt-2">Local account authentication for your security assessment workspace.</p>

    <form className="space-y-3 mt-6" onSubmit={e=>{e.preventDefault();void submitAuth();}}>

     <label className="block text-xs text-slate-400">Email address<input type="email" required value={authEmail} onChange={e=>setAuthEmail(e.target.value)} className="mt-2 w-full rounded-lg bg-slate-950 border border-slate-700 px-3 py-3 text-sm outline-none focus:border-emerald-500"/></label>

     <label className="block text-xs text-slate-400">Password<input type="password" required minLength={authMode==="register"?10:1} value={authPassword} onChange={e=>setAuthPassword(e.target.value)} className="mt-2 w-full rounded-lg bg-slate-950 border border-slate-700 px-3 py-3 text-sm outline-none focus:border-emerald-500"/>{authMode==="register"&&<span className="text-[10px] text-slate-600">Use at least 10 characters.</span>}</label>

     <button className="btn btn-primary w-full justify-center" disabled={authBusy}>{authBusy?"Please wait…":authMode==="login"?"Sign in":"Create account"}</button>

    </form>

    <button className="text-xs text-emerald-400 mt-5" onClick={()=>setAuthMode(authMode==="login"?"register":"login")}>{authMode==="login"?"New here? Create an account":"Already registered? Sign in"}</button>

    {banner&&<p className="text-xs text-rose-300 mt-4 break-words">{banner}</p>}

    <p className="text-[10px] text-slate-600 mt-6">Demo note: accounts are stored in the local SQLite database on this machine. This is not a multi-user production identity system.</p>

   </div>

  </div>;

 }

 return <div className="min-h-screen bg-slate-950 text-slate-200 flex">

  <aside className="w-64 shrink-0 border-r border-slate-800 bg-slate-950 flex flex-col">

   <div className="p-5 border-b border-slate-800"><div className="flex items-center gap-3"><div className="rounded-xl bg-emerald-500/15 p-2 text-emerald-400"><ShieldCheck size={24}/></div><div><div className="font-bold tracking-wide text-white">SENTINEL<span className="text-emerald-400"> AI</span></div><div className="text-[10px] text-slate-500 tracking-[.22em]">APPLICATION DEFENSE</div></div></div></div>

   <div className="px-3 py-5 space-y-1">{nav.map(item=><button key={item.id} onClick={()=>setTab(item.id)} className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg text-left text-sm ${tab===item.id?"bg-emerald-500/10 text-emerald-300 border border-emerald-500/20":"text-slate-400 hover:bg-slate-900 hover:text-white"}`}><item.icon size={17}/>{item.label}</button>)}</div>

   <div className="mt-auto p-4"><div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3"><div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold"><Activity size={14}/> LOCAL SCANNER</div><p className="text-[11px] text-slate-500 mt-2">Static analysis runs in your browser. Gemini is optional.</p><div className="mt-3 flex justify-between text-[10px] text-slate-500"><span>ENGINE STATUS</span><span className="text-emerald-400">READY</span></div></div></div>

  </aside>

  <main className="min-w-0 flex-1 flex flex-col">

   <header className="h-16 border-b border-slate-800 flex items-center justify-between px-7 bg-slate-950/90"><div><div className="text-sm font-semibold text-white">{nav.find(n=>n.id===tab)?.label}</div><div className="text-[11px] text-slate-500">Defensive source review · passive configuration audit</div></div><div className="flex items-center gap-3"><span className={`hidden md:flex items-center gap-2 text-[11px] ${mlStatus.available?"text-emerald-400":"text-amber-400"}`} title={mlStatus.message}><span className={`w-2 h-2 rounded-full ${mlStatus.available?"bg-emerald-400":"bg-amber-400"}`}/>{mlStatus.available?"TRAINED ML ONLINE":"ML SERVICE OFFLINE"}</span><span className="hidden lg:inline text-[10px] text-slate-500">{signedInEmail||"Signed in"}</span><button className="btn" onClick={signOut}><LogOut size={14}/> Sign out</button><button className="btn" onClick={loadSample}><Sparkles size={14}/> Load sample</button></div></header>

   {banner&&<div className="mx-7 mt-4 rounded-lg border border-cyan-900/70 bg-cyan-950/40 px-4 py-3 text-xs text-cyan-200 flex items-start justify-between gap-3"><span>{banner}</span><button onClick={()=>setBanner("")}><X size={14}/></button></div>}

   <div className="p-7 flex-1 min-w-0">

    {tab==="dashboard"&&<section className="space-y-6">

     <div className="flex flex-wrap items-end justify-between gap-3"><div><h1 className="text-2xl font-bold text-white">Security Overview</h1><p className="text-sm text-slate-500 mt-1">Project posture and prioritized code-level risks.</p><p className={`text-[11px] mt-2 ${mlStatus.available?"text-emerald-400":"text-amber-400"}`}>● {mlStatus.available?`${mlStatus.algorithm} · ${mlStatus.datasetRows??"?"} training examples`:"Trained ML service offline — static rules remain available"}</p></div><div className="flex gap-2"><button className="btn" onClick={downloadReport}><Download size={14}/> Export report</button><button className="btn btn-primary" onClick={runVerification} disabled={busy||!findings.length}><Sparkles size={14}/> {busy?"Analyzing…":"Verify with Gemini"}</button></div></div>


     <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <div className="panel p-5"><div className="text-xs text-slate-500">Security Score</div><div className="flex items-end gap-2 mt-2"><span className="text-4xl font-bold text-white">{securityScore}</span><span className="text-sm text-slate-500">/100</span></div><div className="mt-3 h-2 rounded-full bg-slate-800 overflow-hidden"><div className="h-full rounded-full bg-emerald-500 transition-all" style={{width:securityScore+"%"}}/></div><div className="mt-2 text-xs text-emerald-300">{files.length?scoreLabel:"No scan yet"}</div><p className="text-[10px] text-slate-600 mt-1">Critical −25 · High −15 · Medium −8 · Low −3</p></div>
      <div className="panel p-5"><div className="text-xs text-slate-500">Latest Scan</div><div className="text-2xl font-bold text-white mt-2">{scanHistory.length?new Date(scanHistory[0].date).toLocaleString():"No history"}</div><div className="text-xs text-slate-400 mt-2">{scanHistory.length?scanHistory[0].files+" files · "+scanHistory[0].total+" findings":"Scan a project to create history"}</div></div>
      <div className="panel p-5"><div className="text-xs text-slate-500">Scan Comparison</div>{scanHistory.length>1?(()=>{const latest=scanHistory[0],prev=scanHistory[1],delta=latest.total-prev.total,scoreDelta=latest.score-prev.score;return <><div className={"text-2xl font-bold mt-2 "+(delta<0?"text-emerald-300":delta>0?"text-rose-300":"text-slate-200")}>{delta>0?"+":""}{delta} findings</div><div className="text-xs mt-2 text-slate-400">vs previous scan · score {scoreDelta>0?"+":""}{scoreDelta}</div><div className="text-[10px] mt-1">{delta<0?"Improvement detected":delta>0?"More findings detected":"No finding-count change"}</div></>:<div className="text-xs text-slate-400 mt-3">Run at least two scans to compare.</div>}</div>
     </div>
     <div className="panel p-5"><div className="flex items-center justify-between mb-4"><h2 className="font-semibold text-white text-sm">Persistent Scan History</h2><span className="text-[10px] text-slate-500">Saved in this browser · latest 25</span></div>{scanHistory.length?<div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead className="text-[10px] uppercase text-slate-500 border-b border-slate-800"><tr><th className="py-2 pr-3">Date</th><th className="py-2 pr-3">Project</th><th className="py-2 pr-3">Files</th><th className="py-2 pr-3">Findings</th><th className="py-2 pr-3">Severity (C/H/M/L)</th><th>Score</th></tr></thead><tbody className="divide-y divide-slate-800">{scanHistory.map((h:any)=><tr key={h.id}><td className="py-3 pr-3 text-slate-400">{new Date(h.date).toLocaleString()}</td><td className="py-3 pr-3 text-slate-200">{h.project}</td><td className="py-3 pr-3">{h.files}</td><td className="py-3 pr-3">{h.total}</td><td className="py-3 pr-3"><span className="text-rose-300">{h.critical}</span>/<span className="text-orange-300">{h.high}</span>/<span className="text-amber-300">{h.medium}</span>/<span className="text-cyan-300">{h.low}</span></td><td><span className="text-emerald-300 font-semibold">{h.score}/100</span></td></tr>)}</tbody></table><button className="btn mt-3" onClick={()=>{setScanHistory([]);localStorage.removeItem("sentinel_scan_history");setBanner("Scan history cleared.");}}>Clear history</button></div>:<Empty text="Scan a project to start your persistent history."/></div>

     <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">{[{label:"Files Scanned",value:files.length,icon:FileCode2,color:"text-cyan-400"},{label:"Total Findings",value:findings.length,icon:AlertTriangle,color:"text-amber-400"},{label:"Critical & High",value:findings.filter(f=>f.severity==="Critical"||f.severity==="High").length,icon:ShieldCheck,color:"text-red-400"},{label:"Patched Files",value:patchedFiles.size,icon:CheckCircle2,color:"text-emerald-400"}].map(k=><div className="panel p-5" key={k.label}><div className="flex justify-between"><span className="text-xs text-slate-500">{k.label}</span><k.icon size={17} className={k.color}/></div><div className="text-3xl font-bold text-white mt-3">{fmt(k.value)}</div><div className="text-[10px] text-slate-600 mt-2">CURRENT PROJECT SESSION</div></div>)}</div>

     <div className="grid grid-cols-1 xl:grid-cols-5 gap-4"><div className="panel p-5 xl:col-span-3"><h2 className="font-semibold text-white text-sm">Severity distribution</h2><div className="h-64 mt-4">{findings.length?<ResponsiveContainer width="100%" height="100%"><BarChart data={counts}><XAxis dataKey="name" stroke="#64748b" tickLine={false}/><YAxis stroke="#64748b" allowDecimals={false} tickLine={false}/><Tooltip contentStyle={{background:"#0f172a",border:"1px solid #334155",borderRadius:8}}/><Bar dataKey="value" radius={[6,6,0,0]}>{counts.map(c=><Cell key={c.name} fill={c.fill}/>)}</Bar></BarChart></ResponsiveContainer>:<Empty text={emptyNote}/>}</div></div><div className="panel p-5 xl:col-span-2"><h2 className="font-semibold text-white text-sm">Severity breakdown</h2><div className="h-64">{findings.length?<ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={counts.filter(c=>c.value)} dataKey="value" nameKey="name" innerRadius={58} outerRadius={90} paddingAngle={4}>{counts.filter(c=>c.value).map(c=><Cell key={c.name} fill={c.fill}/>)}</Pie><Tooltip contentStyle={{background:"#0f172a",border:"1px solid #334155",borderRadius:8}}/></PieChart></ResponsiveContainer>:<Empty text="No severity data yet."/>}</div><div className="flex flex-wrap gap-3 justify-center">{counts.map(c=><span key={c.name} className="text-[11px] text-slate-400 flex items-center gap-1.5"><i className="w-2 h-2 rounded-full" style={{background:c.fill}}/>{c.name}: {c.value}</span>)}</div></div></div>

     <div className="panel p-5"><div className="flex justify-between items-center mb-4"><h2 className="font-semibold text-white text-sm">File-level risk summary</h2><button className="text-xs text-emerald-400" onClick={()=>setTab("findings")}>Open findings <ChevronRight size={13} className="inline"/></button></div>{fileRisk.length?<div className="space-y-3">{fileRisk.map(f=><div key={f.path} className="flex items-center gap-4 text-xs"><code className="w-1/3 truncate text-slate-300">{f.path}</code><div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden"><div className="h-full bg-emerald-500" style={{width:`${Math.min(100,f.count/Math.max(1,findings.length)*100)}%`}}/></div><span className="w-20 text-right text-slate-400">{f.count} findings</span><span className="w-16 text-right text-orange-300">{f.critical} H/C</span></div>)}</div>:<Empty text={emptyNote}/>}</div>

    </section>}

    {tab==="scanner"&&<section className="space-y-5">

     <div><h1 className="text-2xl font-bold text-white">Project & Endpoint Scanner</h1><p className="text-sm text-slate-500 mt-1">Upload source code for local, non-destructive static analysis.</p></div>

     <div className="grid grid-cols-1 lg:grid-cols-2 gap-4"><div className="panel p-5"><div className="flex items-center gap-2 text-white font-semibold"><FolderOpen size={18} className="text-emerald-400"/> Project source upload</div><p className="text-xs text-slate-500 mt-2">Folder upload preserves relative paths. Supported: Python, JS/TS, JSX/TSX, PHP, Java, HTML, ENV, JSON, YAML.</p><div className="mt-5 grid sm:grid-cols-2 gap-3"><button className="btn btn-primary justify-center" onClick={()=>folderRef.current?.click()}><Upload size={15}/> Choose project folder</button><button className="btn justify-center" onClick={()=>zipRef.current?.click()}><FileArchive size={15}/> Upload ZIP archive</button></div><input ref={folderRef} hidden type="file" multiple {...({webkitdirectory:"",directory:""} as any)} onChange={e=>void folderUpload(e.target.files)}/><input ref={zipRef} hidden type="file" accept=".zip,application/zip" onChange={e=>void zipUpload(e.target.files?.[0])}/><button className="btn w-full justify-center mt-3" onClick={loadSample}><Sparkles size={15}/> Load Sample Vulnerable Project (4 files)</button><div className="mt-5 border-t border-slate-800 pt-4 flex items-center justify-between text-xs"><span className="text-slate-500">{files.length} files · {findings.length} candidates</span><button className="btn" onClick={runVerification} disabled={busy||!findings.length}><Play size={13}/>{busy?"Working…":"Gemini verify"}</button></div></div>

     <div className="panel p-5"><div className="flex items-center gap-2 text-white font-semibold"><Github size={18} className="text-cyan-400"/> GitHub repository import</div><p className="text-xs text-slate-500 mt-2">Import a public GitHub repository and scan supported source files. Private repositories are not supported in this local demo.</p><div className="mt-4 flex gap-2"><input value={githubUrl} onChange={e=>setGithubUrl(e.target.value)} placeholder="https\://github.com/owner/repository" className="min-w-0 flex-1 rounded-lg bg-slate-950 border border-slate-700 px-3 py-2 text-xs outline-none focus:border-cyan-500"/><button className="btn btn-primary" onClick={()=>void importGithub()} disabled={githubBusy||!githubUrl.trim()}><Github size={14}/>{githubBusy?"Importing…":"Import & Scan"}</button></div><p className="text-[10px] text-amber-300 mt-3">Only scan repositories you own or are authorized to assess. Public GitHub API limits apply.</p></div>

      <div className="panel p-5"><div className="flex items-center gap-2 text-white font-semibold"><Globe size={18} className="text-cyan-400"/> Passive HTTP header inspector</div><p className="text-xs text-slate-500 mt-2">Paste raw response headers or fetch a public CORS-enabled endpoint. No payloads are sent.</p><div className="mt-3 flex gap-2"><input value={endpoint} onChange={e=>setEndpoint(e.target.value)} placeholder="https\://your-authorized-site.example" className="min-w-0 flex-1 rounded-lg bg-slate-950 border border-slate-700 px-3 py-2 text-xs outline-none focus:border-cyan-500"/><button className="btn" onClick={inspectUrl} disabled={busy}><Globe size={14}/> Inspect</button></div><label className="flex items-center gap-2 text-[11px] text-slate-400 mt-3"><input type="checkbox" checked={authorized} onChange={e=>setAuthorized(e.target.checked)}/> I own this endpoint or have explicit authorization to inspect it.</label><textarea value={rawHeaders} onChange={e=>setRawHeaders(e.target.value)} placeholder={"HTTP/1.1 200 OK\nContent-Type: text/html\n..."} className="mt-3 w-full h-28 rounded-lg bg-slate-950 border border-slate-700 p-3 text-xs font-mono outline-none focus:border-cyan-500"/><button className="btn btn-primary mt-2" onClick={inspectRaw}><Lock size={14}/> Audit pasted headers</button></div></div>

     {headerChecks.length>0&&<div className="panel p-5"><h2 className="font-semibold text-white text-sm mb-3">Header & cookie audit results</h2><div className="grid md:grid-cols-2 gap-2">{headerChecks.map(c=><div key={c.name} className="rounded-lg bg-slate-950 border border-slate-800 p-3 flex justify-between gap-3"><div><div className="text-xs text-white">{c.name}</div><div className="text-[11px] text-slate-500 mt-1">{c.detail}</div></div><span className={`badge sev-${c.severity} h-fit`}>{c.present?"Present":"Review"}</span></div>)}</div></div>}

     <div className="panel p-5"><h2 className="text-sm font-semibold text-white mb-3">Loaded project files</h2>{files.length?<div className="divide-y divide-slate-800">{files.map(f=><div key={f.path} className="py-2 flex items-center justify-between text-xs"><span className="flex items-center gap-2 text-slate-300"><FileCode2 size={14} className="text-cyan-400"/>{f.path}</span><span className="text-slate-500">{fileCounts(f.path)} findings</span></div>)}</div>:<Empty text={emptyNote}/>}</div>

    </section>}

    {tab==="findings"&&<section className="space-y-5">

     <div className="flex flex-wrap items-end justify-between gap-3"><div><h1 className="text-2xl font-bold text-white">Findings Explorer</h1><p className="text-sm text-slate-500 mt-1">Every location uses 1-based source line numbering.</p></div><button className="btn" onClick={runVerification} disabled={busy||!findings.length}><Sparkles size={14}/> Verify candidates</button></div>

     <div className="panel p-4"><div className="flex flex-wrap gap-2"><div className="relative flex-1 min-w-48"><Search size={14} className="absolute left-3 top-3 text-slate-500"/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search file, rule, category…" className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-xs outline-none focus:border-emerald-500"/></div><select value={severityFilter} onChange={e=>setSeverityFilter(e.target.value)} className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs"><option>All</option>{severityOrder.map(s=><option key={s}>{s}</option>)}</select></div>

     <div className="overflow-x-auto mt-4"><table className="w-full text-left text-xs"><thead className="text-[10px] uppercase tracking-wider text-slate-500 border-b border-slate-800"><tr><th className="py-3 pr-4">Severity</th><th className="py-3 pr-4">File : Line</th><th className="py-3 pr-4">Finding</th><th className="py-3 pr-4">Confidence</th><th className="py-3">Source</th></tr></thead><tbody className="divide-y divide-slate-800">{visibleFindings.map(f=><tr key={f.id} onClick={()=>setSelected(f)} className={`cursor-pointer hover:bg-slate-800/60 ${selected?.id===f.id?"bg-slate-800/50":""}`}><td className="py-3 pr-4"><span className={`badge sev-${f.severity}`}>{f.severity}</span></td><td className="py-3 pr-4"><button className="font-mono text-cyan-300 hover:text-cyan-200 text-left">{f.filePath} : Line {f.lineNumber}</button></td><td className="py-3 pr-4"><div className="text-slate-200">{f.title}</div><div className="text-[10px] text-slate-500 mt-1">{f.cweId||f.predictedCategory} {f.isTruePositive===false?"· Gemini marked false positive":""}</div></td><td className="py-3 pr-4 text-slate-300">{f.mlModelConfidence!==undefined?f.mlModelConfidence:f.mlConfidence}%<div className={`text-[10px] mt-1 ${f.mlPrediction==="Vulnerable"?"text-orange-300":f.mlPrediction==="Review Required"?"text-amber-300":f.mlPrediction==="Likely Safe"?"text-emerald-300":"text-slate-600"}`}>{f.mlPrediction||"Heuristic fallback (not trained)"}</div></td><td className="py-3 text-slate-500">{f.detectionSource}</td></tr>)}</tbody></table>{!visibleFindings.length&&<Empty text={emptyNote}/>}</div></div>

     {selected&&<div className="panel p-5 grid xl:grid-cols-2 gap-5"><div><div className="flex items-center justify-between"><h2 className="font-semibold text-white text-sm">Code inspector</h2><span className={`badge sev-${selected.severity}`}>{selected.severity}</span></div><div className="mt-3 rounded-lg bg-slate-950 border border-slate-800 p-3 overflow-auto text-xs font-mono">{selected.codeSnippet.split("\n").map((line,i)=>{const num=Number(line.match(/^\d+:/)?.[0].replace(":",""));return <span key={i} className={`code-line ${num===selected.lineNumber?"active":""}`}>{line}</span>})}</div><div className="flex gap-3 mt-3 text-xs text-slate-400"><span>Trained ML score: <b className="text-cyan-300">{selected.mlModelConfidence!==undefined?`${selected.mlModelConfidence}%`:"Not connected"}</b></span><span>Prediction: <b className="text-slate-200">{selected.mlPrediction||"Not scored"}</b></span><span>Rule: {selected.ruleId}</span></div></div><div><div className="flex items-center justify-between"><h2 className="font-semibold text-white text-sm">AI explanation & remediation</h2><button className="btn btn-primary" onClick={()=>void explainSelectedFinding()} disabled={explainBusy}><Sparkles size={13}/>{explainBusy?"Explaining…":"Explain with AI"}</button></div><div className="mt-3 rounded-lg bg-slate-950 border border-slate-800 p-4 text-xs"><div className="flex gap-2 flex-wrap mb-3"><span className="badge bg-slate-800 text-slate-300">{selected.cweId||"CWE pending"}</span><span className="badge bg-slate-800 text-slate-300">{selected.owaspCategory||"OWASP pending"}</span><span className={`badge ${selected.isTruePositive===undefined?"bg-slate-800 text-slate-400":selected.isTruePositive?"bg-red-950 text-red-300":"bg-emerald-950 text-emerald-300"}`}>{selected.isTruePositive===undefined?"Not verified":selected.isTruePositive?"True positive":"False positive"}</span></div><p className="text-slate-300 leading-relaxed">{selected.explanation||"Run Gemini verification to obtain a contextual explanation and secure replacement for this candidate."}</p>{selected.fixedLineSnippet&&<div className="mt-4"><div className="text-[10px] uppercase text-emerald-400 mb-2">Suggested exact-line replacement</div><pre className="whitespace-pre-wrap text-emerald-200 bg-emerald-950/20 border border-emerald-900/50 rounded p-3">{selected.fixedLineSnippet}</pre></div>}<button className="btn mt-3" onClick={()=>{setActiveFile(selected.filePath);setTab("chat");}}><Sparkles size={13}/> Open full-file fixer</button></div></div></div>}

    </section>}

    {tab==="chat"&&<section className="space-y-4">

     <div><h1 className="text-2xl font-bold text-white">Gemini AI Security Chat</h1><p className="text-sm text-slate-500 mt-1">Context-aware full-file secure remediation with rollback snapshots.</p></div>

     <div className="grid grid-cols-1 xl:grid-cols-5 gap-4 min-h-[650px]">

      <div className="panel p-4 xl:col-span-3 flex flex-col min-h-[650px]"><div className="flex items-center justify-between border-b border-slate-800 pb-3"><div className="flex items-center gap-2 text-sm font-semibold"><MessageSquare size={16} className="text-emerald-400"/> Security assistant</div><span className="text-[10px] text-slate-500">gemini-2.5-pro</span></div>

       <div className="flex flex-wrap gap-2 py-3">{["Fix all vulnerabilities & give full corrected file","Explain line-by-line security risks","Harden input validation & error handling"].map(p=><button key={p} className="text-[10px] border border-slate-700 rounded-full px-3 py-2 text-slate-400 hover:text-emerald-300 hover:border-emerald-700" onClick={()=>void sendChat(p)}>{p}</button>)}</div>

       <div className="flex-1 overflow-y-auto space-y-3 pr-1">{chatMessages.length?chatMessages.map((m,i)=><div key={i} className={`rounded-xl p-3 text-xs leading-relaxed ${m.role==="user"?"bg-emerald-950/40 border border-emerald-900/60 ml-8":"bg-slate-950 border border-slate-800 mr-5"}`}><div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">{m.role==="user"?"You":"Gemini"}</div><div className="whitespace-pre-wrap break-words">{m.text}</div>{m.code&&<button className="btn mt-3" onClick={()=>void copy(m.code!)}><Copy size={12}/> Copy full code</button>}</div>):<Empty text="Choose a target file and ask Gemini to review or remediate the complete file."/>}{chatBusy&&<div className="text-xs text-cyan-300 animate-pulse">Gemini is reviewing the complete file…</div>}</div>

       <div className="border-t border-slate-800 pt-3 flex gap-2"><textarea value={chatInput} onChange={e=>setChatInput(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();void sendChat();}}} placeholder="Ask for a secure full-file remediation…" rows={2} className="flex-1 resize-none rounded-lg bg-slate-950 border border-slate-700 p-3 text-xs outline-none focus:border-emerald-500"/><button className="btn btn-primary self-end" onClick={()=>void sendChat()} disabled={chatBusy||!chatInput.trim()}><Send size={14}/></button></div>

      </div>

      <div className="panel p-4 xl:col-span-2 flex flex-col gap-4"><div><label className="text-xs text-slate-400 block mb-2">Target file selector</label><select value={currentFile?.path||""} onChange={e=>{setActiveFile(e.target.value);setGeneratedCode("");}} className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-xs">{files.map(f=><option key={f.path} value={f.path}>{f.path} ({fileCounts(f.path)} findings)</option>)}</select></div>

       {currentFile?<><div className="rounded-lg bg-slate-950 border border-slate-800 p-3"><div className="flex justify-between text-xs mb-2"><span className="text-slate-300">File context</span><span className="text-slate-500">{currentFile.content.split("\n").length} lines</span></div><pre className="text-[10px] text-slate-500 max-h-48 overflow-auto whitespace-pre-wrap">{currentFile.content}</pre><div className="mt-2 text-[10px] text-cyan-300">Findings: {findings.filter(f=>f.filePath===currentFile.path).map(f=>`Line ${f.lineNumber}`).join(", ")||"None detected"}</div></div>

       <div className="flex-1 min-h-32 rounded-lg border border-emerald-900/50 bg-emerald-950/10 p-3"><div className="flex items-center justify-between mb-2"><span className="text-xs font-semibold text-emerald-300">Generated full-file replacement</span>{generatedCode&&<button className="text-[10px] text-emerald-300" onClick={()=>void copy(generatedCode)}><Copy size={12} className="inline"/> Copy Full Code</button>}</div>{generatedCode?<pre className="text-[10px] max-h-52 overflow-auto whitespace-pre-wrap text-emerald-100">{generatedCode}</pre>:<div className="text-[11px] text-slate-600">Generated code appears here after Gemini responds with a fenced full-file block.</div>}</div>

       <button className="btn btn-primary justify-center" disabled={!generatedCode} onClick={applyCode}><CheckCircle2 size={14}/> Apply Full Code to Project Workspace</button></>:<Empty text="Load a project to select a target file."/>}

       <div className="border-t border-slate-800 pt-3"><div className="flex items-center gap-2 text-xs font-semibold text-white mb-2"><RotateCcw size={14} className="text-amber-400"/> Version snapshots & rollback</div>{snapshots.length?snapshots.map(s=><div key={s.id} className="flex items-center justify-between gap-2 py-2 border-b border-slate-800 text-[10px]"><div className="min-w-0"><div className="truncate text-slate-300">{s.filePath}</div><div className="text-slate-600">{new Date(s.createdAt).toLocaleString()}</div></div><button className="btn !py-1 !px-2" onClick={()=>rollback(s)}><RotateCcw size={11}/> Rollback</button></div>):<div className="text-[10px] text-slate-600">No patch snapshots yet.</div>}</div>

      </div>

     </div>

    </section>}

   </div>

   <footer className="border-t border-slate-800 px-7 py-3 text-[10px] text-slate-600 flex justify-between"><span>Sentinel AI · Defensive assessment workspace</span><span>{files.length} files · {findings.length} findings · {patchedFiles.size} patched</span></footer>

  </main>

 </div>;

}

function Empty({text}:{text:string}){return <div className="h-full min-h-24 flex flex-col items-center justify-center text-center text-xs text-slate-600 gap-2"><ShieldCheck size={22} className="text-slate-700"/>{text}</div>}

export default App;