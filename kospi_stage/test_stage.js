const fs=require('fs'); const S=require('./stage.js'); if(process.argv[2]) S.P.confirm=+process.argv[2];
const L=fs.readFileSync('data/kospi_daily.csv','utf8').trim().split('\n').slice(1).map(r=>r.split(','));
const D={d:L.map(r=>r[0]),o:L.map(r=>+r[1]),h:L.map(r=>+r[2]),l:L.map(r=>+r[3]),c:L.map(r=>+r[4]),v:L.map(r=>+r[5])};
const R=S.classify(D); const seg=S.segments(D,R.stage);
console.log('segments',seg.length);
seg.forEach(s=>console.log(s.stage,s.from,s.to,s.days,(s.ret*100).toFixed(1)+'%'));
const n=D.c.length; const x=R.X[n-1];
console.log('now',R.stage[n-1],'cand',R.cand[n-1],JSON.stringify(x,(k,v)=>typeof v==='number'?+v.toFixed(4):v));
for(const h of [20,60,120]){const st={};for(let i=0;i<n-h;i++){const s=R.stage[i];if(!s)continue;(st[s]=st[s]||[]).push(D.c[i+h]/D.c[i]-1);}
console.log('fwd',h,Object.entries(st).map(([k,a])=>k+': n'+a.length+' avg '+(a.reduce((p,q)=>p+q,0)/a.length*100).toFixed(1)+'% win '+(a.filter(z=>z>0).length/a.length*100).toFixed(0)+'%').join(' | '));}
