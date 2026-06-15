// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Amar Akram

/* NetzPilot — Browser-Forecast-Kern (reines JS, kein Backend).
 *
 * Bildet die Methode der Python-Engine treu, aber vereinfacht ab:
 *   - Baseline: saisonal-naiv  yhat(d,h) = y(d-7,h)
 *   - Korrektur: pro Stunde geschrumpfter jüngster Drift gegenüber Vorwoche
 *     corr(h) = s * mean_{letzte K Tage} ( y(t,h) - y(t-7,h) ),  s in {0,.25,.5,.75,1} auf Holdout
 *   - Bänder: split-conformal aus Kalibrierfenster, finite-sample-Quantil (1-alpha)(1+1/n)
 *   - Bewertung: rolling-origin über die letzten n_test Tage
 *
 * Ehrlicher Browser-Schnellschätzer. Die belastbaren Zahlen liefert die volle
 * Python-Engine (ShrunkCorrector + rolling CQR) via scripts/pilot_in_a_box.py.
 */
'use strict';

const WEEKDAY_RE = /[\s,;]+[A-Za-zÄÖÜäöüß.]{2,3}\.?\s*$/;
const LOAD_HINTS = /(last|load|mw|kw|wirk|verbrauch|leistung|menge|summe|differenz|netz|dba|wert|value|p_?ges|p\s*\(kw\))/i;

function splitLines(text){ return text.replace(/\r\n?/g,'\n').split('\n').filter(l=>l.trim().length); }
function detectSep(h){ const c=(h.match(/;/g)||[]).length,k=(h.match(/,/g)||[]).length,t=(h.match(/\t/g)||[]).length; if(t>=c&&t>=k)return '\t'; return c>=k?';':','; }

function parseNum(raw){
  if(raw==null) return NaN;
  let s=String(raw).trim(); if(!s) return NaN;
  if(/\d,\d/.test(s)) s=s.replace(/\./g,'').replace(',', '.');       // DE-Dezimalkomma
  else s=s.replace(/(?<=\d)\.(?=\d{3}\b)/g,'');                        // Tausenderpunkt
  const v=parseFloat(s); return isFinite(v)?v:NaN;
}
function parseDate(raw){
  if(raw==null) return null;
  let s=String(raw).trim().replace(WEEKDAY_RE,'');
  let m=s.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/)||s.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if(m) return {y:+m[1],mo:+m[2],d:+m[3],h:m[4]!=null?+m[4]:0};
  m=s.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})(?:[ T](\d{1,2}):(\d{2}))?/);
  if(m) return {y:+m[3],mo:+m[2],d:+m[1],h:m[4]!=null?+m[4]:0};
  const dt=new Date(s); if(!isNaN(dt)) return {y:dt.getFullYear(),mo:dt.getMonth()+1,d:dt.getDate(),h:dt.getHours()};
  return null;
}
function isIndexLike(vals){
  const x=vals.filter(isFinite); if(x.length<5) return false;
  const step=x[1]-x[0]; if(Math.abs(step)<1) return false;
  for(let i=1;i<x.length;i++) if(Math.abs((x[i]-x[i-1])-step)>1e-9) return false;
  return true;
}
function mean(a){let s=0,n=0;for(const v of a)if(isFinite(v)){s+=v;n++;}return n?s/n:NaN;}
function std(a){const m=mean(a);let s=0,n=0;for(const v of a)if(isFinite(v)){s+=(v-m)*(v-m);n++;}return n?Math.sqrt(s/n):0;}
function quantile(sorted,q){ if(!sorted.length)return NaN; const p=(sorted.length-1)*Math.max(0,Math.min(1,q)); const lo=Math.floor(p),hi=Math.ceil(p); return lo===hi?sorted[lo]:sorted[lo]+(sorted[hi]-sorted[lo])*(p-lo); }
function mae(a,b){let s=0,n=0;for(let i=0;i<a.length;i++)if(isFinite(a[i])&&isFinite(b[i])){s+=Math.abs(a[i]-b[i]);n++;}return n?s/n:NaN;}

function loadCsv(text, opt){
  opt=opt||{};
  const lines=splitLines(text); if(lines.length<3) throw new Error('CSV zu kurz.');
  const sep=detectSep(lines[0]);
  const header=lines[0].split(sep).map(h=>h.trim().replace(/^﻿/,''));
  const rows=lines.slice(1).map(l=>l.split(sep));
  const ncol=header.length, col=j=>rows.map(r=>r[j]);

  let tsIdx=opt.tsCol!=null?header.indexOf(opt.tsCol):-1;
  if(tsIdx<0){ let best=-1,bs=-1; for(let j=0;j<ncol;j++){ const s=col(j); let ok=0; for(const v of s) if(parseDate(v)) ok++; const sc=ok/s.length; if(sc>bs){bs=sc;best=j;} } if(bs<0.5) throw new Error('Keine Zeitstempel-Spalte erkannt.'); tsIdx=best; }

  let loadIdx=opt.loadCol!=null?header.indexOf(opt.loadCol):-1;
  if(loadIdx<0){
    const cand=[];
    for(let j=0;j<ncol;j++){ if(j===tsIdx)continue; const nums=col(j).map(parseNum); const ok=nums.filter(isFinite).length/nums.length;
      if(ok>0.5 && !isIndexLike(nums) && std(nums.filter(isFinite))>0) cand.push({j,uniq:new Set(nums.filter(isFinite)).size}); }
    if(!cand.length) throw new Error('Keine plausible Lastspalte gefunden.');
    const named=cand.filter(c=>LOAD_HINTS.test(header[c.j])); const pool=named.length?named:cand;
    pool.sort((a,b)=>b.uniq-a.uniq); loadIdx=pool[0].j;
  }

  const tsRaw=col(tsIdx), loadRaw=col(loadIdx);
  const buckets=new Map();
  for(let i=0;i<rows.length;i++){ const dt=parseDate(tsRaw[i]); const v=parseNum(loadRaw[i]); if(!dt||!isFinite(v))continue;
    const key=`${dt.y}-${String(dt.mo).padStart(2,'0')}-${String(dt.d).padStart(2,'0')} ${String(dt.h).padStart(2,'0')}`;
    let b=buckets.get(key); if(!b){b={s:0,n:0,y:dt.y,mo:dt.mo,d:dt.d,h:dt.h};buckets.set(key,b);} b.s+=v; b.n++; }
  const unit=(opt.unit||'MW').toLowerCase(); const scale=unit==='kw'?1e-3:unit==='w'?1e-6:1;
  const keys=[...buckets.keys()].sort();
  const hourly=keys.map(k=>{const b=buckets.get(k);return {y:b.y,mo:b.mo,d:b.d,h:b.h,val:(b.s/b.n)*scale};});
  return {hourly, tsCol:header[tsIdx], loadCol:header[loadIdx], unit:opt.unit||'MW'};
}

function toDaily(hourly){
  const byDay=new Map();
  for(const r of hourly){ const dk=`${r.y}-${String(r.mo).padStart(2,'0')}-${String(r.d).padStart(2,'0')}`;
    let a=byDay.get(dk); if(!a){a=new Array(24).fill(null);byDay.set(dk,a);} a[r.h]=r.val; }
  const dates=[...byDay.keys()].sort(); const load2d=[],days=[];
  for(const dk of dates){ const a=byDay.get(dk); if(a.every(v=>v!=null&&isFinite(v))){load2d.push(a);days.push(dk);} }
  return {load2d,days};
}

function evaluate(load2d, days, cfg){
  cfg=cfg||{}; const K=cfg.K||28, HOLD=cfg.hold||7, CAL=cfg.cal||21, nTest=cfg.nTest||14;
  const n=load2d.length; const S=[0,0.25,0.5,0.75,1.0];
  if(n<7+HOLD+nTest+1) throw new Error(`Zu wenig vollständige Tage (${n}). Mind. ~${7+HOLD+nTest+1} nötig.`);

  // Punktprognose für Tag d (nutzt nur Daten < d) -> {pred[24], s}
  // Margin-Guard: Korrektur (s>0) nur wenn sie das reine saisonal-naiv (s=0) auf dem
  // Holdout um >0.5 % schlägt — verhindert Overfitting des kurzen Holdouts (sonst Skill < 0).
  function predictDay(d){
    const corr=new Array(24).fill(0);
    for(let h=0;h<24;h++){ const diffs=[]; for(let t=d-1;t>=7&&diffs.length<K;t--) diffs.push(load2d[t][h]-load2d[t-7][h]); corr[h]=mean(diffs)||0; }
    const errAt=s=>{ const ae=[]; for(let t=Math.max(7,d-HOLD);t<d;t++) for(let h=0;h<24;h++) ae.push(Math.abs(load2d[t][h]-(load2d[t-7][h]+s*corr[h]))); return mean(ae); };
    const e0=errAt(0); let bestS=0,bestErr=e0;
    for(const s of S){ if(s===0) continue; const e=errAt(s); if(e<bestErr){bestErr=e;bestS=s;} }
    if(bestS>0 && !(bestErr < e0*0.995)) bestS=0;   // nur bei klarer Verbesserung korrigieren
    return {pred:load2d[d-7].map((b,h)=>b+bestS*corr[h]), s:bestS};
  }
  // split-conformal Band um pred[d], pro Stunde, aus Kalibrierfenster [d-CAL, d)
  function band(d,pred,alpha){
    const lo=new Array(24),hi=new Array(24);
    const calPreds=[]; for(let t=Math.max(8,d-CAL);t<d;t++) calPreds.push({t,p:predictDay(t).pred});
    for(let h=0;h<24;h++){ const res=calPreds.map(o=>Math.abs(load2d[o.t][h]-o.p[h])).sort((a,b)=>a-b);
      const q=quantile(res,Math.min(1,(1-alpha)*(1+1/res.length))); lo[h]=pred[h]-q; hi[h]=pred[h]+q; }
    return {lo,hi};
  }

  const A=[],M=[],SN=[],PE=[],L80=[],H80=[],L90=[],H90=[]; let lastDay=null;
  for(let d=n-nTest; d<n; d++){
    const {pred}=predictDay(d); const b80=band(d,pred,0.2), b90=band(d,pred,0.1);
    const act=load2d[d], sn=load2d[d-7], pe=load2d[d-1];
    for(let h=0;h<24;h++){ A.push(act[h]);M.push(pred[h]);SN.push(sn[h]);PE.push(pe[h]);L80.push(b80.lo[h]);H80.push(b80.hi[h]);L90.push(b90.lo[h]);H90.push(b90.hi[h]); }
    if(d===n-1) lastDay={date:days[d],act,pred,lo80:b80.lo,hi80:b80.hi,lo90:b90.lo,hi90:b90.hi};
  }
  const maeM=mae(A,M),maeSn=mae(A,SN),maePe=mae(A,PE);
  let ape=0,na=0; for(let i=0;i<A.length;i++) if(Math.abs(A[i])>1e-9){ape+=Math.abs((A[i]-M[i])/A[i]);na++;}
  const cov=(lo,hi)=>{let c=0;for(let i=0;i<A.length;i++) if(A[i]>=lo[i]&&A[i]<=hi[i])c++; return 100*c/A.length;};
  return { nDays:n, nTest, meanLoad:mean(A), MAE:maeM, MAPE:na?100*ape/na:NaN, MASE:maeM/maeSn,
    skillSnaive:100*(1-maeM/maeSn), skillPersist:100*(1-maeM/maePe), cov80:cov(L80,H80), cov90:cov(L90,H90), lastDay };
}

if(typeof module!=='undefined'&&module.exports){ module.exports={loadCsv,toDaily,evaluate,parseDate,parseNum}; }
