import type { HeaderCheck } from "../types";

export function inspectHeaders(raw: string): HeaderCheck[] {
 const normalized = raw.replace(/\r/g,"");
 const headers = new Map<string,string[]>();
 for (const line of normalized.split("\n")) {
  const match = line.match(/^\s*([^:]+):\s*(.*)$/);
  if (match) headers.set(match[1].trim().toLowerCase(),[...(headers.get(match[1].trim().toLowerCase())||[]),match[2].trim()]);
 }
 const checks: Array<[string,string,string]> = [
  ["content-security-policy","Content-Security-Policy","A policy helps constrain script, style, and resource origins."],
  ["strict-transport-security","Strict-Transport-Security","Enforces HTTPS for supported browsers."],
  ["x-frame-options","X-Frame-Options","Helps prevent clickjacking; CSP frame-ancestors is the modern control."],
  ["x-content-type-options","X-Content-Type-Options","Use nosniff to reduce MIME-sniffing risks."],
  ["referrer-policy","Referrer-Policy","Controls how much referrer information is sent."]
 ];
 const result: HeaderCheck[] = checks.map(([key,name,detail])=>{
  const value=headers.get(key)?.join("; ");
  return {name,present:!!value,status:value||"Missing",severity:value?"Low":"Normal",detail:value?`${detail} Observed: ${value}`:detail};
 });
 const cookies=headers.get("set-cookie")||[];
 if(cookies.length) for(const cookie of cookies) {
  for(const [flag,sev] of [["HttpOnly","High"],["Secure","High"],["SameSite","Normal"]] as const) {
   const present=new RegExp(`(?:^|;\\s*)${flag}(?:=|;|$)`,"i").test(cookie);
   result.push({name:`Cookie ${flag}`,present,status:present?"Present":"Missing",severity:present?"Low":sev,detail:present?`Cookie includes ${flag}.`:`Set-Cookie is missing ${flag}: ${cookie.split(";")[0]}.`});
  }
 }
 return result;
}

export async function inspectEndpoint(url: string): Promise<string> {
 const parsed = new URL(url);
 if (!["http:","https:"].includes(parsed.protocol)) throw new Error("Only HTTP and HTTPS endpoints are supported.");
 const response = await fetch(parsed.toString(), {method:"GET",mode:"cors",credentials:"omit",redirect:"follow"});
 const rows: string[] = [`HTTP ${response.status} ${response.statusText}`];
 response.headers.forEach((value,key)=>rows.push(`${key}: ${value}`));
 return rows.join("\n");
}