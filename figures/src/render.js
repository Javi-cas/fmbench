const {chromium}=require('playwright'); const fs=require('fs');
(async()=>{const [svg,png,pdf,scale]=process.argv.slice(2); const b=await chromium.launch();
const s=fs.readFileSync(svg,'utf8'); const m=s.match(/viewBox="0 0 (\d+) (\d+)"/); const w=+m[1],h=+m[2];
const p=await b.newPage({viewport:{width:w,height:h},deviceScaleFactor:+scale||2});
await p.setContent(`<html><body style="margin:0">${s}</body></html>`); await p.waitForTimeout(300);
await p.screenshot({path:png,clip:{x:0,y:0,width:w,height:h}});
if(pdf){await p.pdf({path:pdf,width:w+'px',height:h+'px',printBackground:true,pageRanges:'1'});}
await b.close();})();
