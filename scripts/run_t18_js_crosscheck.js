'use strict';

const fs = require('fs');
const path = require('path');
const { loadCsv, toDaily, evaluate } = require('./forecast_engine_core.js');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'data_cache', 't18_js_crosscheck');

const CASES = [
  {
    slug: 'bitterfeld_jhl_ns_2024',
    csv: 'data_cache/real/bitterfeld_jhl_ns_2024.csv',
    pilot: 'data_cache/pilot/t18_bitterfeld_jhl_ns_2024/pilot_metrics.json',
    loadCol: 'Wert',
    unit: 'kW',
    region: 'ST',
  },
  {
    slug: 'bitterfeld_jhl_msns_2024',
    csv: 'data_cache/real/bitterfeld_jhl_msns_2024.csv',
    pilot: 'data_cache/pilot/t18_bitterfeld_jhl_msns_2024/pilot_metrics.json',
    loadCol: 'Wert',
    unit: 'kW',
    region: 'ST',
  },
  {
    slug: 'neuruppin_na_ms_2022',
    csv: 'data_cache/real/neuruppin_lgl_strom_2022.csv',
    pilot: 'data_cache/pilot/t18_neuruppin_na_ms_2022/pilot_metrics.json',
    loadCol: 'Wert.11',
    unit: 'kW',
    region: 'BB',
  },
  {
    slug: 'neuruppin_na_ns_2022',
    csv: 'data_cache/real/neuruppin_lgl_strom_2022.csv',
    pilot: 'data_cache/pilot/t18_neuruppin_na_ns_2022/pilot_metrics.json',
    loadCol: 'Wert.13',
    unit: 'kW',
    region: 'BB',
  },
  {
    slug: 'waren_bezug_vnb_ms_2025',
    csv: 'data_cache/real/waren_2026_03_27_LGL_Strom_2025_Waren.csv',
    pilot: 'data_cache/pilot/t18_waren_bezug_vnb_ms_2025/pilot_metrics.json',
    loadCol: 'Wert.2',
    unit: 'kW',
    region: 'MV',
  },
  {
    slug: 'waren_na_ms_2025',
    csv: 'data_cache/real/waren_2026_03_27_LGL_Strom_2025_Waren.csv',
    pilot: 'data_cache/pilot/t18_waren_na_ms_2025/pilot_metrics.json',
    loadCol: 'Wert.10',
    unit: 'kW',
    region: 'MV',
  },
  {
    slug: 'waren_differenzbilanzierung_2025',
    csv: 'data_cache/real/waren_2026_03_31_LGL-Strom_2025_Waren_p12.csv',
    pilot: 'data_cache/pilot/t18_waren_differenzbilanzierung_2025/pilot_metrics.json',
    loadCol: 'Wert',
    unit: 'kW',
    region: 'MV',
  },
];

function detectSep(line) {
  const counts = [';', '\t', ','].map(sep => [sep, (line.match(new RegExp(sep === '\t' ? '\\t' : sep, 'g')) || []).length]);
  counts.sort((a, b) => b[1] - a[1]);
  return counts[0][0];
}

function withMangledDuplicateHeaders(text) {
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  const headerIdx = lines.findIndex(line => /^\s*Datum[;\t,]von[;\t,]bis\b/i.test(line));
  if (headerIdx < 0) return text;
  const sep = detectSep(lines[headerIdx]);
  const seen = new Map();
  const cells = lines[headerIdx].split(sep).map(cell => {
    const trimmed = cell.trim().replace(/^\uFEFF/, '');
    if (!trimmed) return cell;
    const count = seen.get(trimmed) || 0;
    seen.set(trimmed, count + 1);
    return count === 0 ? trimmed : `${trimmed}.${count}`;
  });
  lines[headerIdx] = cells.join(sep);
  return lines.join('\n');
}

function r1(value) {
  return Number.isFinite(value) ? Math.round(value * 10) / 10 : null;
}

function close(a, b, tol) {
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tol;
}

function runCase(item) {
  const text = withMangledDuplicateHeaders(fs.readFileSync(path.join(root, item.csv), 'utf8'));
  const parsed = loadCsv(text, { unit: item.unit, loadCol: item.loadCol });
  const daily = toDaily(parsed.hourly);
  const load2d = daily.load2d.slice(-120);
  const days = daily.days.slice(-120);
  const js = evaluate(load2d, days, { nTest: 14, region: item.region });
  const py = JSON.parse(fs.readFileSync(path.join(root, item.pilot), 'utf8'));
  const comparison = {
    MAE_MW: { python: py.MAE_MW, js: r1(js.MAE), match: close(py.MAE_MW, r1(js.MAE), 0.1) },
    MAPE_pct: { python: py['MAPE_%'], js: r1(js.MAPE), match: close(py['MAPE_%'], r1(js.MAPE), 0.1) },
    skill_snaive_pct: { python: py['skill_vs_snaive_%'], js: r1(js.skillSnaive), match: close(py['skill_vs_snaive_%'], r1(js.skillSnaive), 0.2) },
    skill_persist_pct: { python: py['skill_vs_persistenz_%'], js: r1(js.skillPersist), match: close(py['skill_vs_persistenz_%'], r1(js.skillPersist), 0.2) },
  };
  return {
    slug: item.slug,
    csv: item.csv,
    load_col: item.loadCol,
    region: item.region,
    js_days_used: js.nDays,
    js_n_test_days: js.nTest,
    parsed_ts_col: parsed.tsCol,
    parsed_load_col: parsed.loadCol,
    comparison,
    js_confirmed: Object.values(comparison).every(row => row.match),
  };
}

fs.mkdirSync(outDir, { recursive: true });
const results = CASES.map(runCase);
for (const result of results) {
  fs.writeFileSync(path.join(outDir, `${result.slug}.json`), JSON.stringify(result, null, 2));
}
fs.writeFileSync(path.join(outDir, 'summary.json'), JSON.stringify({
  created_at: new Date().toISOString(),
  method: 'Node forecast_engine_core.js with duplicate-header mangling only; load columns remain explicit.',
  results,
}, null, 2));
console.log(JSON.stringify(results, null, 2));
