/* _cockpit_script_check.js — rev5 (2026-06-04): liest den <script>-Body LIVE aus cockpit.html
   und eval't ihn gegen den DOM-Stub. Damit kann der Harness NIE mehr von der UI divergieren
   (rev2–rev4 trugen eine eingebettete Kopie, die bei jeder UI-Aenderung nachgezogen werden musste).
   Lauf aus dem Repo-Root:  node scripts/_cockpit_script_check.js
   Pfad-Override (z. B. Sandbox):  COCKPIT_PATH=/pfad/zu/cockpit.html node scripts/_cockpit_script_check.js */
console.log('REV5-EVAL-START');
const fs=require('fs');

/* ---------- DOM-Stub ---------- */
function el(id){return {id,innerHTML:'',textContent:'',className:'',value:'',checked:false,
  style:{},dataset:{},files:[],onclick:null,onchange:null,
  lastElementChild:{textContent:''},
  addEventListener(){},setAttribute(){},getBoundingClientRect(){return{left:0,width:980}},
  querySelectorAll(){return []},
  click(){this._clicked=(this._clicked||0)+1}}}
const S={};
const document={getElementById:id=>S[id]||(S[id]=el(id)),documentElement:{dataset:{}},
  createElement(tag){const e=el('created-'+tag);CREATED.push(e);return e}};
const CREATED=[];
const window={innerWidth:1400};
const URL={createObjectURL:b=>'blob:fake',revokeObjectURL(){}};
class Blob{constructor(parts,opts){this.parts=parts;this.opts=opts}}
const fetch=()=>Promise.reject(new Error('offline (Stub)'));
const location={protocol:'http:'};   // healthFailText unterscheidet file:// von Server-down
let LASTFD=null;
const FormData=class{constructor(){this.entries=[];LASTFD=this}append(k,v){this.entries.push([k,v])}};

/* ---------- Cockpit-Skript LIVE laden + eval ---------- */
const SRC_PATH=process.env.COCKPIT_PATH||'netzpilot/service/cockpit.html';
const SRC=fs.readFileSync(SRC_PATH,'utf8');
const parts=SRC.split('<script>');
let NC=0,FAIL=0;
function check(ok,msg){NC++;if(!ok){FAIL++;console.log('FAIL '+NC+': '+msg)}else console.log('ok   '+NC+': '+msg)}
check(parts.length===2&&parts[1].includes('</script>'),'cockpit.html hat genau einen <script>-Block ('+SRC_PATH+')');
if(parts.length!==2){console.log('ABBRUCH');process.exit(1)}
eval(parts[1].split('</script>')[0]);   // definiert render, resultFromJson, historyChipsHtml, … im Modul-Scope
// Lazy vivifizierte Elemente fuer die Checks anlegen (das Skript fasst z. B. csvFile erst beim Klick an):
['csvFile','hist','utility','kDate','chart','intraday','horizon','track','error','tip','tBadge','horizonDays','horizonBands','idActuals','holAdd','holRemove'].forEach(i=>document.getElementById(i));

/* ---------- Checks ---------- */
const H24=f=>Array.from({length:24},(_,h)=>f(h));
const MOCK={utility:'Mock Stadtwerk',unit:'MW',forecast_date:'2026-06-04',
  forecast:H24(h=>({hour:h,p10:30+5*Math.sin(h/24*6.28),p50:35+6*Math.sin(h/24*6.28),p90:41+7*Math.sin(h/24*6.28)})),
  residual_forecast:{forecast:H24(h=>({hour:h,p50:20+4*Math.sin(h/24*6.28)}))},
  residual_feedback:{rho:0.31},
  input_validation:{enabled:true,quality_score:0.973},
  drift:{status:'ok',needs_recalibration:false},
  congestion:{window_hours:[18,19],basis:'load'},
  asset_limit:{rating_kw:43000,source:'einheitlich',note:'Eine Rating-Wahrheit. <script>alert(1)</script>'},
  redispatch:{hourly:H24(h=>({hour:h,intervention:h===18||h===19,shed_kw:h===18?120:h===19?80:0})),saved_vs_naive_kwh:42.5,grid_safe:true},
  tariff_schedule:{saving_eur:1234,feasible:true,caveat:'Tarif-Caveat'},
  dispatch_plan:{newsvendor_saving_eur:2345,grid_safe:true,risk_averse:{beta:0.6,risk_cvar_delta_vs_newsvendor_eur:-99}},
  overload:{max_exceedance_prob:0.03,hours_at_risk:2,caveat:'Overload-Caveat'},
  hosting_capacity:{hosting_capacity_kw:5200,binding_hour:19},
  thermal:{max_exceedance_prob:0.01,expected_loss_of_life_h_total:0.42},
  economics_realized:{status:'ok',annualized_saving_eur:9500,prob_positive:0.87},
  economics_expected:{expected_eur_per_year:9500},
  mmm:{status:'available',abs_volumen_reduktion_mwh:12.345,abs_volumen_reduktion_at_price_eur:610},
  pool_dispatch:{grid_safe:true,all_feasible:true,pool_shed_kwh:0.123,
    hourly:H24(h=>({hour:h,pool_demand_kw:900+10*h,pool_limit_kw:1000,pool_shed_kw:h===19?5:0}))},
  track_record:{chain_ok:true,n_realized:3,n_pending:1,n_forecasts_stored:4,
    aggregate:{mae_mean:1.234,coverage_mean_pct:79.2},
    last_30_days:[{target_date:'2026-06-01',issued_at_utc:'2026-05-31T09:00:00Z',mae:1.3,bias:0.2,coverage_p10_p90_pct:83.3},
                  {target_date:'2026-06-02',issued_at_utc:'2026-06-01T09:00:00Z',mae:1.1,bias:-0.1,coverage_p10_p90_pct:79.2},
                  {target_date:'2026-06-03',issued_at_utc:'2026-06-02T09:00:00Z',mae:1.3,bias:0.0,coverage_p10_p90_pct:75.0}]},
  horizon:{issued_after:'2026-06-03',bands_mode:'k1',bands:'k=1 voll; k>=2 nur P50',bands_note:'k=1 voll; k>=2 nur P50',
    days:[{date:'2026-06-05',horizon:2,hours:H24(h=>({hour:h,p50:36+h/10}))},
          {date:'2026-06-06',horizon:3,hours:H24(h=>({hour:h,p50:34+h/12}))}]},
  intraday:{applied:true,update_hour:12,n_hours_used:12,delta_mw:1.234,
    hours_rest:H24(h=>({hour:h,p50:36+2*Math.sin(h/24*6.28)})).slice(12),
    caveat:'Gemessen auf 3 echten Reihen; nicht jeder Tag gewinnt.'}};
const MOCK_PH={...MOCK,horizon:{issued_after:'2026-06-03',bands_mode:'per_horizon',
  bands:'alle Horizonte mit kalibriertem Band: k=1 = Produktionsband; k>=2 = 1-Schritt-Stundenform mal s_k + CQR-c_k.',
  days:[{date:'2026-06-05',horizon:2,band:{scale:1.25,conf_c:0.4,n_cal_days:28},
    hours:H24(h=>({hour:h,p10:32+h/10,p50:36+h/10,p90:41+h/10}))},
        {date:'2026-06-06',horizon:3,band:{scale:1.37,conf_c:0.6,n_cal_days:28},
    hours:H24(h=>({hour:h,p10:30+h/12,p50:34+h/12,p90:40+h/12}))}]}};

(async()=>{
  check(typeof render==='function'&&typeof resultFromJson==='function','Skript ge-eval-t: render+resultFromJson definiert');
  render(null);
  check(S.kDate.textContent==='–'&&S.chart.innerHTML===''&&S.intraday.innerHTML===''&&S.horizon.innerHTML===''&&S.track.innerHTML==='','render(null) leert alles');
  render(MOCK);
  check(S.kDate.textContent==='2026-06-04','KPI Datum');
  check(/\d+\.\d MW/.test(S.kPeak.textContent),'KPI Peak in MW');
  check(S.chart.innerHTML.includes('<svg')&&S.chart.innerHTML.includes('var(--band)'),'Hero-Chart mit Band');
  check(S.chart.innerHTML.includes('stroke="var(--warn)"')&&S.chart.innerHTML.includes('stroke-dasharray="5 5"')&&S.intraday.innerHTML.includes('Intraday δ='),'Intraday: Resttag-Linie + Badge');
  render({...MOCK,intraday:{applied:false,update_hour:1,n_hours_used:0,delta_mw:0,hours_rest:[],reason:'zu wenig valide Ist-Stunden',caveat:'Caveat bleibt sichtbar'}});
  check(S.intraday.innerHTML.includes('zu wenig valide')&&S.intraday.innerHTML.includes('Caveat bleibt sichtbar'),'Intraday: applied=false zeigt reason + Caveat');
  render(MOCK);
  check(S.chart.innerHTML.includes('stroke-dasharray="6 4"'),'Residual-Linie gezeichnet');
  check(S.chart.innerHTML.includes('opacity=".12"'),'Engpassfenster schattiert');
  check(S.ctlStrip.innerHTML.includes('var(--warn)')&&S.ctlStrip.innerHTML.split('class="cell"').length===25,'§14a-Streifen: 24 Zellen, Eingriff markiert');
  check(S.control.innerHTML.includes('18, 19 h'),'Engpassfenster im Control-Panel');
  check(S.gridRisk.innerHTML.includes('kW · Quelle'),'Netz-Panel Rating-Zeile');
  check(S.money.innerHTML.includes('€'),'€-Panel gefüllt');
  check(S.pool.innerHTML.includes('<i title='),'Pool-Minibars');
  check(S.horizon.innerHTML.includes('D+2')&&S.horizon.innerHTML.includes('MWh')&&S.horizon.innerHTML.includes('nur P50'),'Horizont-Tabelle: D+2/D+3 P50');
  check(!S.horizon.innerHTML.includes('>s=')&&!S.horizon.innerHTML.includes('Band-Felder'),'k1-Horizont: nur P50 ohne scale/warn');
  render(MOCK_PH);
  check(S.horizon.innerHTML.includes('P10-P90 (Peakstunde)')&&S.horizon.innerHTML.includes('s=1.25'),'per_horizon: Band-Spalte + scale-Label');
  check(!S.horizon.innerHTML.includes('Band-Felder')&&S.horizon.innerHTML.includes('alle Horizonte mit kalibriertem Band'),'per_horizon: kein Warn-Badge, Note sichtbar');
  render({...MOCK,horizon:{...MOCK_PH.horizon,bands_mode:'k1',bands:'k=1 voll; k>=2 nur P50'}});
  check(S.horizon.innerHTML.includes('Band-Felder'),'k1-Modus warnt, falls Bandfelder auftauchen');
  check(S.tBadge.textContent==='Hash-Kette OK'&&S.track.innerHTML.includes('Zieltag'),'Track-Record-Tabelle + Badge');
  check(S.caveats.innerHTML.includes('&lt;script&gt;')&&!S.caveats.innerHTML.includes('<script>alert'),'esc(): HTML im JSON wird escaped');
  S.chart.innerHTML='';S.themeBtn.onclick();
  check(document.documentElement.dataset.theme==='dark'&&S.chart.innerHTML.includes('<svg'),'Theme-Toggle + Re-Render');
  hover({clientX:500,clientY:100});
  check(S.tip.innerHTML.includes('P50')&&S.tip.style.display==='block','Hover-Tooltip gefüllt');
  check(resultFromJson(JSON.stringify(MOCK),'m.json').utility==='Mock Stadtwerk','resultFromJson akzeptiert Ergebnis-JSON');
  let t1=0;try{resultFromJson(JSON.stringify({date:'x',hours:[1,2]}),'forecast_next_day.json')}catch(e){t1=/kein NetzPilot-Ergebnis-JSON/.test(e.message)}
  check(t1===true,'Fremd-JSON wird mit erklärender Meldung abgelehnt');
  let t2=0;try{resultFromJson('{kaputt','x.json')}catch(e){t2=/kein gültiges JSON/.test(e.message)}
  check(t2===true,'kaputtes JSON → klare Meldung');
  const p1='data_cache/service_store/Mein_Stadtwerk/latest.json';
  const p2='data_cache/service_store/Mein_Stadtwerk/2026-01-01.json';
  const p3='data_cache/forecast_next_day.json';
  for(const p of [p1,p2,p3])check(fs.existsSync(p),'echte Datei vorhanden: '+p);
  const mkEvt=(path,name)=>({target:{files:[{name,text:async()=>fs.readFileSync(path,'utf8')}],value:'C:\\fake\\'+name}});
  let evt=mkEvt(p3,'forecast_next_day.json');await S.jsonFile.onchange(evt);
  check(/kein NetzPilot-Ergebnis-JSON/.test(S.error.textContent),'Upload Fremd-JSON → Fehlertext im UI');
  check(evt.target.value==='','Datei-Input wird zurückgesetzt');
  evt=mkEvt(p1,'latest.json');await S.jsonFile.onchange(evt);
  check(S.error.textContent===''&&S.kDate.textContent!=='–','Upload echtes latest.json → rendert');
  check(S.chart.innerHTML.includes('<svg'),'latest.json: Hero-Chart gezeichnet');
  evt=mkEvt(p2,'2026-01-01.json');await S.jsonFile.onchange(evt);
  check(S.error.textContent===''&&S.kDate.textContent==='2026-01-01','Upload echtes 2026-01-01.json → rendert');
  S.saveJson.onclick();
  const a=CREATED[CREATED.length-1];
  check(a&&a._clicked===1&&a.href==='blob:fake'&&/^netzpilot_.*2026-01-01\.json$/.test(a.download),'JSON speichern → Download-Anker');
  render(null);S.saveJson.onclick();
  check(/Nichts zu speichern/.test(S.error.textContent),'JSON speichern ohne Daten → Meldung');
  S.horizonDays.value='3';
  S.horizonBands.value='per_horizon';
  S.csvFile.files=[{name:'lastgang.xlsx'}];
  await S.runLive.onclick();
  check(LASTFD&&LASTFD.entries.some(e=>e[0]==='file')&&!LASTFD.entries.some(e=>e[0]==='csv_path'),'Live-Lauf mit Datei → multipart "file"');
  check(LASTFD.entries.some(e=>e[0]==='horizon_days'&&e[1]==='3'),'Live-Lauf sendet horizon_days');
  check(LASTFD.entries.some(e=>e[0]==='horizon_bands'&&e[1]==='per_horizon'),'Live-Lauf sendet horizon_bands');
  check(/offline/.test(S.error.textContent),'Live-Lauf-Fehler (Stub) im UI');
  S.csvFile.files=[];
  await S.runLive.onclick();
  check(LASTFD.entries.some(e=>e[0]==='csv_path')&&!LASTFD.entries.some(e=>e[0]==='file'),'Live-Lauf ohne Datei → csv_path');
  S.utility.value='Mock Stadtwerk';S.idActuals.value='21.0,22.5';
  await S.runIntraday.onclick();
  check(LASTFD.entries.some(e=>e[0]==='utility'&&e[1]==='Mock Stadtwerk')&&LASTFD.entries.some(e=>e[0]==='actuals'&&e[1]==='21.0,22.5'),'Intraday-POST sendet utility+actuals');
  const chips=historyChipsHtml(['2026-01-01','2026-01-02'],5);
  check(chips.includes('5 gespeicherte Tage')&&chips.split('class="chip"').length===3&&chips.includes('data-d="2026-01-02"'),'historyChipsHtml: Chips + Gesamtzahl');
  check(historyChipsHtml([],0)===''&&historyChipsHtml(null,0)==='','historyChipsHtml: leer → kein HTML');
  S.hist.innerHTML='ALT';await refreshHistory();
  check(S.hist.innerHTML==='','refreshHistory offline → still geleert');
  await loadDate('Mein Stadtwerk','2026-01-01');
  check(/offline/.test(S.error.textContent),'loadDate offline → Fehlertext');
  S.hist.innerHTML='x';render(MOCK);S.clear.onclick();
  check(S.kDate.textContent==='–'&&S.hist.innerHTML==='','Leeren räumt auch den Verlauf');
  /* rev5: Mandanten-datalist */
  /* Feiertags-Override: Caveats-Anzeige + FormData */
  render({...MOCK,holiday_overrides:{added:['2026-05-15'],removed:[],target_is_holiday:true,caveat:'Nutzer-Annahme'}});
  check(S.caveats.innerHTML.includes('Feiertags-Override')&&S.caveats.innerHTML.includes('2026-05-15')&&S.caveats.innerHTML.includes('Nutzer-Annahme'),'Feiertags-Override im Caveats-Register sichtbar');
  S.holAdd.value=' 2026-05-15 ';S.holRemove.value='';
  await S.runLive.onclick();
  check(LASTFD.entries.some(e=>e[0]==='holiday_add'&&e[1]==='2026-05-15')&&!LASTFD.entries.some(e=>e[0]==='holiday_remove'),'Live-Lauf sendet holiday_add (getrimmt), remove nur wenn gesetzt');
  S.holAdd.value='';
  /* Stale-Server-Diagnose: nacktes 404 = Route fehlt = alter Prozess */
  err(new Error('Not Found'));
  check(/älteren Version|Start_NetzPilot/.test(S.error.textContent),'err("Not Found") → Neustart-Hinweis statt Rätsel');
  err(new Error("Keine Prognose für 'X' gespeichert."));
  check(/Keine Prognose/.test(S.error.textContent),'echte 404-Detailtexte bleiben unverändert');
  /* Health-Diagnose: file:// vs Server-down */
  check(/Als Datei geöffnet/.test(healthFailText('file:'))&&/127\.0\.0\.1:8000\/cockpit/.test(healthFailText('file:')),'healthFailText(file:) erklärt richtigen Aufruf');
  check(/Start_NetzPilot\.bat/.test(healthFailText('http:')),'healthFailText(http:) verweist auf Server-Start');
  const opts=utilityOptionsHtml(['Mein Stadtwerk','A<&>B']);
  check(opts.split('<option ').length===3&&opts.includes('value="A&lt;&amp;&gt;B"'),'utilityOptionsHtml: Optionen + escaped');
  check(utilityOptionsHtml([])===''&&utilityOptionsHtml(null)==='','utilityOptionsHtml: leer → kein HTML');
  /* Server-Dateiauswahl */
  const fopts=fileOptionsHtml(['a.csv','b<x>.xlsx']);
  check(fopts.split('<option').length===4&&fopts.includes('value="b&lt;x&gt;.xlsx"')&&fopts.startsWith('<option value="">'),'fileOptionsHtml: Platzhalter + Optionen + escaped');
  check(fileOptionsHtml([]).split('<option').length===2,'fileOptionsHtml: leer → nur Platzhalter');
  /* Blind-Challenge */
  const CMOCK={source_file:'kunde.csv',n_days_history:364,n_test:84,mean_load_mw:29.2,
    mae_model_mw:1.21,mae_snaive_mw:1.49,mae_persist_mw:2.37,mape_pct:3.51,
    vs_snaive:{skill_pct:18.7,ci95:[10.2,26.8],significant_5pct:true,p_model_better_pct:100,days_won_pct:67.9},
    vs_persist:{skill_pct:49.0,ci95:[37.7,63.7],significant_5pct:true},method:'Rolling-Origin + Block-Bootstrap'};
  renderChallenge(CMOCK);
  check(S.challenge.innerHTML.includes('+18.7')&&S.challenge.innerHTML.includes('signifikant (5')&&S.challenge.innerHTML.includes('kunde.csv'),'renderChallenge: Verdikt + Datei + sig-Badge');
  check(S.challenge.innerHTML.includes('1.21')&&S.challenge.innerHTML.includes('Persistenz'),'renderChallenge: MAE-Vergleich');
  renderChallenge({...CMOCK,vs_snaive:{...CMOCK.vs_snaive,significant_5pct:false,skill_pct:2.1,ci95:[-1.0,5.2]}});
  check(S.challenge.innerHTML.includes('nicht signifikant'),'renderChallenge: n.s.-Variante ehrlich');
  renderChallenge(null);
  check(S.challenge.innerHTML==='','renderChallenge(null) leert');
  await S.runChallenge.onclick();
  check(/offline/.test(S.error.textContent)&&S.challenge.innerHTML==='','runChallenge offline → Fehler im UI, Panel geleert');
  console.log(FAIL===0?`ALLE ${NC} CHECKS GRÜN`:`${FAIL}/${NC} CHECKS ROT`);
  console.log('REV5-EVAL-END');
  process.exit(FAIL===0?0:1);
})();
