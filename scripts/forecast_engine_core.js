/* NetzPilot — ENGINE-TREUER Forecast-Kern in JS (repliziert die Python-Engine 1:1).
 *
 * Quellen (exakt nachgebaut):
 *   netzpilot/features/build.py  -> build_features (13 reine Abweichungs-Features, KEINE Rohlevel),
 *                                   base = load[d-7], resid_target = load[d]-load[d-7], first=8
 *   netzpilot/models/ridge_correction.py -> RidgeCorrector (standardisiert, λ, b0=mean(y))
 *   netzpilot/models/robust_corrector.py -> ShrunkCorrector (Tail-Shrink s∈{0,.25,.5,.75,1}, refit auf allen)
 *   netzpilot/eval/backtest.py   -> rolling_origin; P10/P90 = In-Sample-Residuenquantile je Stunde
 *
 * Reines JS, läuft im Browser. Für Referenz/Signifikanz weiterhin scripts/pilot_in_a_box.py.
 */
'use strict';

const WEEKDAY_RE=/[\s,;]+[A-Za-zÄÖÜäöüß.]{2,3}\.?\s*$/;
const LOAD_HINTS=/(last|load|mw|kw|wirk|verbrauch|leistung|menge|summe|differenz|netz|dba|wert|value|p_?ges|p\s*\(kw\))/i;
const splitLines=t=>t.replace(/\r\n?/g,'\n').split('\n').filter(l=>l.trim().length);
function detectSep(h){const c=(h.match(/;/g)||[]).length,k=(h.match(/,/g)||[]).length,t=(h.match(/\t/g)||[]).length;if(t>=c&&t>=k)return '\t';return c>=k?';':',';}
function parseNum(raw){if(raw==null)return NaN;let s=String(raw).trim();if(!s)return NaN;if(/\d,\d/.test(s))s=s.replace(/\./g,'').replace(',', '.');else s=s.replace(/(?<=\d)\.(?=\d{3}\b)/g,'');const v=parseFloat(s);return isFinite(v)?v:NaN;}
function parseDate(raw){if(raw==null)return null;let s=String(raw).trim().replace(WEEKDAY_RE,'');let m=s.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/)||s.match(/^(\d{4})-(\d{2})-(\d{2})$/);if(m)return{y:+m[1],mo:+m[2],d:+m[3],h:m[4]!=null?+m[4]:0};m=s.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})(?:[ T](\d{1,2}):(\d{2}))?/);if(m){const yy=+m[3];return{y:yy<100?2000+yy:yy,mo:+m[2],d:+m[1],h:m[4]!=null?+m[4]:0};}const dt=new Date(s);if(!isNaN(dt))return{y:dt.getFullYear(),mo:dt.getMonth()+1,d:dt.getDate(),h:dt.getHours()};return null;}
function isIndexLike(v){const x=v.filter(isFinite);if(x.length<5)return false;const st=x[1]-x[0];if(Math.abs(st)<1)return false;for(let i=1;i<x.length;i++)if(Math.abs((x[i]-x[i-1])-st)>1e-9)return false;return true;}
const mean=a=>{let s=0,n=0;for(const v of a)if(isFinite(v)){s+=v;n++;}return n?s/n:NaN;};
const std=a=>{const m=mean(a);let s=0,n=0;for(const v of a)if(isFinite(v)){s+=(v-m)*(v-m);n++;}return n?Math.sqrt(s/n):0;};
// numpy.quantile linear (gleiche Methode wie Python rolling_origin)
const quantile=(s,q)=>{if(!s.length)return NaN;const p=(s.length-1)*Math.max(0,Math.min(1,q)),lo=Math.floor(p),hi=Math.ceil(p);return lo===hi?s[lo]:s[lo]+(s[hi]-s[lo])*(p-lo);};
const mae=(a,b)=>{let s=0,n=0;for(let i=0;i<a.length;i++)if(isFinite(a[i])&&isFinite(b[i])){s+=Math.abs(a[i]-b[i]);n++;}return n?s/n:NaN;};

/* DE-Feiertage: feste + Karfreitag/Oster-/Pfingstmontag; regionale Extras fuer Pilot-Checks. */
function holidaySet(years,region){const set=new Set();region=(region||'NW').toUpperCase();const fixed=[[1,1],[5,1],[10,3],[12,25],[12,26]];
  const addFixed=(y,m,d)=>set.add(`${y}-${m}-${d}`);
  for(const y of years){for(const [m,d] of fixed)addFixed(y,m,d);
    if(['BW','BY','ST'].includes(region))addFixed(y,1,6);
    if(region==='MV'&&y>=2023)addFixed(y,3,8);
    if(['BB','BW','HB','HH','MV','NI','SH','SN','ST','TH'].includes(region))addFixed(y,10,31);
    if(['BW','BY','NW','RP','SL'].includes(region))addFixed(y,11,1);
    const a=y%19,b=Math.floor(y/100),c=y%100,d2=Math.floor(b/4),e=b%4,f=Math.floor((b+8)/25),g=Math.floor((b-f+1)/3),h=(19*a+b-d2-g+15)%30,i=Math.floor(c/4),k=c%4,l=(32+2*e+2*i-h-k)%7,mm=Math.floor((a+11*h+22*l)/451),month=Math.floor((h+l-7*mm+114)/31),day=((h+l-7*mm+114)%31)+1;
    const E=Date.UTC(y,month-1,day),add=n=>{const t=new Date(E+n*864e5);return `${t.getUTCFullYear()}-${t.getUTCMonth()+1}-${t.getUTCDate()}`;};
    set.add(add(-2));set.add(add(1));set.add(add(50));
    if(['BW','BY','HE','NW','RP','SL'].includes(region))set.add(add(60)); } // Fronleichnam
  return set;}
function dowOf(s){const [y,mo,d]=s.split('-').map(Number);return (new Date(Date.UTC(y,mo-1,d)).getUTCDay()+6)%7;} // Mo=0
function isHol(s,hs){const [y,mo,d]=s.split('-').map(Number);return hs.has(`${y}-${mo}-${d}`)?1:0;}

const TIME_RE=/^\d{1,2}:\d{2}/;
function loadCsv(text,opt){opt=opt||{};const lines=splitLines(text);if(lines.length<3)throw new Error('CSV zu kurz.');
  // Trenner: maximiere Median-Feldzahl (robust gegen Metadaten-Vorspann)
  let sep=';',bestMed=-1;for(const s of [';','\t',',']){const c=lines.slice(0,60).map(l=>l.split(s).length).sort((a,b)=>a-b);const med=c[Math.floor(c.length/2)];if(med>bestMed){bestMed=med;sep=s;}}
  const rows=lines.map(l=>l.split(sep).map(c=>c.trim().replace(/^﻿/,'')));const ncol=Math.max(...rows.map(r=>r.length));
  // Datumsspalte = längster zusammenhängender Lauf parsebarer Datumswerte (überspringt Vorspann)
  let dateCol=-1,runStart=0,runLen=0;
  for(let j=0;j<ncol;j++){let cur=0,cs=0;for(let i=0;i<rows.length;i++){if(rows[i][j]!=null&&parseDate(rows[i][j])){if(cur===0)cs=i;cur++;if(cur>runLen){runLen=cur;runStart=cs;dateCol=j;}}else cur=0;}}
  if(dateCol<0||runLen<48)throw new Error('Keine Zeitstempel-Spalte erkannt.');
  const data=rows.slice(runStart,runStart+runLen);const header=runStart>0?rows[runStart-1]:[];
  // optionale Overrides per Header-Name
  if(opt.tsCol&&header.indexOf(opt.tsCol)>=0)dateCol=header.indexOf(opt.tsCol);
  // separate Uhrzeit-Spalte (HH:MM) erkennen und mit Datum kombinieren
  let timeCol=-1;for(let j=0;j<ncol;j++){if(j===dateCol)continue;let ok=0;for(const r of data)if(r[j]&&TIME_RE.test(r[j]))ok++;if(ok/data.length>0.8){timeCol=j;break;}}
  // Lastspalte
  let loadIdx=opt.loadCol&&header.indexOf(opt.loadCol)>=0?header.indexOf(opt.loadCol):-1;
  if(loadIdx<0){const cand=[];for(let j=0;j<ncol;j++){if(j===dateCol||j===timeCol)continue;const nums=data.map(r=>parseNum(r[j]));const ok=nums.filter(isFinite).length/data.length;if(ok>0.5&&!isIndexLike(nums)&&std(nums.filter(isFinite))>0)cand.push({j,uniq:new Set(nums.filter(isFinite)).size});}
    if(!cand.length)throw new Error('Keine plausible Lastspalte gefunden.');const named=header.length?cand.filter(c=>LOAD_HINTS.test(header[c.j]||'')):[];const pool=named.length?named:cand;pool.sort((a,b)=>b.uniq-a.uniq);loadIdx=pool[0].j;}
  const bk=new Map();
  for(const r of data){let dstr=r[dateCol];if(timeCol>=0&&r[timeCol])dstr=dstr+' '+r[timeCol];const dt=parseDate(dstr);const v=parseNum(r[loadIdx]);if(!dt||!isFinite(v))continue;
    const key=`${dt.y}-${String(dt.mo).padStart(2,'0')}-${String(dt.d).padStart(2,'0')} ${String(dt.h).padStart(2,'0')}`;
    let b=bk.get(key);if(!b){b={s:0,n:0,y:dt.y,mo:dt.mo,d:dt.d,h:dt.h};bk.set(key,b);}b.s+=v;b.n++;}
  const unit=(opt.unit||'MW').toLowerCase(),scale=unit==='kw'?1e-3:unit==='w'?1e-6:1;
  const keys=[...bk.keys()].sort();const hourly=keys.map(k=>{const b=bk.get(k);return{y:b.y,mo:b.mo,d:b.d,h:b.h,val:(b.s/b.n)*scale};});
  return{hourly,tsCol:(header[dateCol]||('Spalte'+dateCol))+(timeCol>=0?'+'+(header[timeCol]||timeCol):''),loadCol:header[loadIdx]||('Spalte'+loadIdx)};}
function toDaily(hourly){const byDay=new Map();for(const r of hourly){const dk=`${r.y}-${String(r.mo).padStart(2,'0')}-${String(r.d).padStart(2,'0')}`;let a=byDay.get(dk);if(!a){a=new Array(24).fill(null);byDay.set(dk,a);}a[r.h]=r.val;}
  const dates=[...byDay.keys()].sort();const load2d=[],days=[];for(const dk of dates){const a=byDay.get(dk);if(a.every(v=>v!=null&&isFinite(v))){load2d.push(a);days.push(dk);}}return{load2d,days};}

/* build_features (13 Features/Stunde) — exakt wie Python build.py */
function featuresForDay(load2d,d,days,hs){
  const dev_prev=new Array(24); for(let h=0;h<24;h++) dev_prev[h]=load2d[d-1][h]-load2d[d-8][h];
  const dev_mean=mean(dev_prev);
  const trend=mean(load2d[d-1])-mean(load2d[d-8]);
  const dow=dowOf(days[d]), wknd=dow>=5?1:0, hol=isHol(days[d],hs);
  const X=[];
  for(let h=0;h<24;h++){
    X.push([1.0, dev_prev[h], dev_mean, trend, load2d[d-1][h]-load2d[d-7][h],
      Math.sin(2*Math.PI*h/24),Math.cos(2*Math.PI*h/24),Math.sin(4*Math.PI*h/24),Math.cos(4*Math.PI*h/24),
      Math.sin(2*Math.PI*dow/7),Math.cos(2*Math.PI*dow/7), wknd, hol]);
  }
  return X;
}

function solveSPD(A,b){const F=b.length,M=A.map((r,i)=>r.concat([b[i]]));
  for(let c=0;c<F;c++){let p=c;for(let r=c+1;r<F;r++)if(Math.abs(M[r][c])>Math.abs(M[p][c]))p=r;[M[c],M[p]]=[M[p],M[c]];const d=M[c][c]||1e-12;
    for(let r=0;r<F;r++){if(r===c)continue;const f=M[r][c]/d;for(let k=c;k<=F;k++)M[r][k]-=f*M[c][k];}}return M.map((r,i)=>r[F]/(r[i]||1e-12));}
/* RidgeCorrector EXAKT wie Python: Spalte0=Intercept(1), nur Spalten1.. standardisiert,
   Xs=[1,(X[:,1:]-mu)/sd], A=Xs'Xs+λI, A[0,0]-=λ (Intercept ungestraft), w=solve(A,Xs'y). */
function ridgeFit(X,y,lam){const n=X.length,F=X[0].length,P=F-1;
  const mu=new Array(P).fill(0),sd=new Array(P).fill(0);
  for(let j=0;j<P;j++){let s=0;for(let i=0;i<n;i++)s+=X[i][j+1];mu[j]=s/n;}
  for(let j=0;j<P;j++){let s=0;for(let i=0;i<n;i++){const d=X[i][j+1]-mu[j];s+=d*d;}sd[j]=Math.sqrt(s/n);if(sd[j]===0)sd[j]=1;}
  const Xs=X.map(r=>{const o=[1.0];for(let j=0;j<P;j++)o.push((r[j+1]-mu[j])/sd[j]);return o;});
  const A=Array.from({length:F},()=>new Array(F).fill(0));
  for(let a=0;a<F;a++)for(let b=0;b<F;b++){let s=0;for(let i=0;i<n;i++)s+=Xs[i][a]*Xs[i][b];A[a][b]=s+(a===b?lam:0);}
  A[0][0]-=lam;                       // Intercept nicht bestrafen
  const bb=new Array(F).fill(0);for(let a=0;a<F;a++){let s=0;for(let i=0;i<n;i++)s+=Xs[i][a]*y[i];bb[a]=s;}
  return{mu,sd,w:solveSPD(A,bb)};}
function ridgePredict(m,X){return X.map(r=>{let s=m.w[0];for(let j=0;j<m.mu.length;j++)s+=((r[j+1]-m.mu[j])/m.sd[j])*m.w[j+1];return s;});}
/* ShrunkCorrector: ntail=min(n-24,max(24,⌊0.2n⌋)) if n>48; s auf Tail; refit auf allen; predict=s*ridge */
function shrunkFit(X,y,lam){const n=X.length;let s=1.0;
  const ntail=n>48?Math.min(n-24,Math.max(24,Math.floor(n*0.2))):0;
  if(ntail>=24){const head=n-ntail;const rt=ridgeFit(X.slice(0,head),y.slice(0,head),lam);const pt=ridgePredict(rt,X.slice(head));const yt=y.slice(head);
    let bE=Infinity;for(const cand of [0,.25,.5,.75,1]){const e=mean(yt.map((v,i)=>Math.abs(v-cand*pt[i])));if(e<bE){bE=e;s=cand;}}}
  return{ridge:ridgeFit(X,y,lam),shrink:s};}
function shrunkPredict(m,X){const p=ridgePredict(m.ridge,X);return p.map(v=>v*m.shrink);}

/* rolling_origin (engine-treu) + In-Sample-Residuenquantile-Bänder */
function evaluate(load2d,days,cfg){cfg=cfg||{};const first=8,nTest=cfg.nTest||28,lam=cfg.lam||10.0,n=load2d.length;
  if(n<first+nTest+1)throw new Error(`Zu wenig vollständige Tage (${n}). Mind. ~${first+nTest+1} nötig.`);
  const hs=holidaySet([...new Set(days.map(d=>+d.slice(0,4)))],cfg.region||'NW');
  const A=[],M=[],SN=[],PE=[],P10=[],P90=[],P05=[],P95=[];let last=null;
  for(let d=n-nTest;d<n;d++){
    // Training [first,d): Features + Residualziel
    const Xtr=[],ytr=[],baseTr=[],actTr=[];
    for(let t=first;t<d;t++){const Xt=featuresForDay(load2d,t,days,hs);for(let h=0;h<24;h++){Xtr.push(Xt[h]);ytr.push(load2d[t][h]-load2d[t-7][h]);baseTr.push(load2d[t-7][h]);actTr.push(load2d[t][h]);}}
    const model=shrunkFit(Xtr,ytr,lam);
    const fitTr=shrunkPredict(model,Xtr);                       // residual prediction
    // In-Sample-Residuen je Stunde -> Quantile
    const resByH=Array.from({length:24},()=>[]);
    for(let i=0;i<Xtr.length;i++){const h=i%24;resByH[h].push(actTr[i]-(baseTr[i]+fitTr[i]));}
    const q=(arr,p)=>{const s=arr.slice().sort((a,b)=>a-b);return quantile(s,p);};
    const q10=resByH.map(a=>q(a,.10)),q90=resByH.map(a=>q(a,.90)),q05=resByH.map(a=>q(a,.05)),q95=resByH.map(a=>q(a,.95));
    // Prognose Tag d
    const corr=shrunkPredict(model,featuresForDay(load2d,d,days,hs));
    const pred=load2d[d-7].map((b,h)=>b+corr[h]);
    const act=load2d[d],sn=load2d[d-7],pe=load2d[d-1];
    for(let h=0;h<24;h++){A.push(act[h]);M.push(pred[h]);SN.push(sn[h]);PE.push(pe[h]);
      P10.push(pred[h]+q10[h]);P90.push(pred[h]+q90[h]);P05.push(pred[h]+q05[h]);P95.push(pred[h]+q95[h]);}
    if(d===n-1)last={date:days[d],act,pred,lo90:q05.map((v,h)=>pred[h]+v),hi90:q95.map((v,h)=>pred[h]+v),lo80:q10.map((v,h)=>pred[h]+v),hi80:q90.map((v,h)=>pred[h]+v)};
  }
  const mM=mae(A,M),mS=mae(A,SN),mP=mae(A,PE);let ap=0,na=0;for(let i=0;i<A.length;i++)if(Math.abs(A[i])>1e-9){ap+=Math.abs((A[i]-M[i])/A[i]);na++;}
  const cov=(lo,hi)=>{let c=0;for(let i=0;i<A.length;i++)if(A[i]>=lo[i]&&A[i]<=hi[i])c++;return 100*c/A.length;};
  return{nDays:n,nTest,meanLoad:mean(A),MAE:mM,MAPE:na?100*ap/na:NaN,MASE:mM/mS,skillSnaive:100*(1-mM/mS),skillPersist:100*(1-mM/mP),
    cov80:cov(P10,P90),cov90:cov(P05,P95),lastDay:last};}

if(typeof module!=='undefined'&&module.exports){module.exports={loadCsv,toDaily,evaluate,featuresForDay,ridgeFit,ridgePredict,shrunkFit,holidaySet};}
/* engine-faithful ridge core — rev3 (multiline-header + split date/time CSV) */
