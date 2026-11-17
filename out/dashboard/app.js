// Backscatter dashboard: renders window.RADAR (written by main.ts as data.js) into the board.
// Static page, no framework, no network: the ranked board with value bands, a detail drawer per
// protocol with its netted farm plan, the walk-forward proof, and the already-trading live set.
'use strict';

const R = window.RADAR;
const main = document.getElementById('board');

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const usd = (n) => (n === null || n === undefined ? '—' : '$' + Math.round(n).toLocaleString('en-US'));
const usdSigned = (n) => (n > 0 ? '+' : n < 0 ? '−' : '') + usd(Math.abs(n)).slice(0) ;
const pct = (n) => (n === null || n === undefined ? '—' : (n > 0 ? '+' : '') + n.toFixed(1) + '%');
const state = { q: '', verdict: 'all', watchOnly: false, sort: 'rank' };

const boardMin = Math.min(...R.rows.map((r) => r.value_low_usd));
const boardMax = Math.max(...R.rows.map((r) => r.value_high_usd));
const logPos = (x) => {
  const lo = Math.log(Math.max(1, boardMin));
  const hi = Math.log(Math.max(1, boardMax));
  return hi === lo ? 50 : ((Math.log(Math.max(1, x)) - lo) / (hi - lo)) * 100;
};
const bandHtml = (r) => {
  const l = logPos(r.value_low_usd);
  const h = logPos(r.value_high_usd);
  const p = logPos(r.implied_value_usd);
  return `<div class="band"><div class="range" style="left:${l.toFixed(1)}%;width:${Math.max(1.5, h - l).toFixed(1)}%"></div>` +
    `<div class="point" style="left:${p.toFixed(1)}%"></div></div>` +
    `<div class="bandlbl">${usd(r.value_low_usd)} – ${usd(r.value_high_usd)}</div>`;
};

function headerHtml() {
  const v = R.validation;
  const vc = R.verdict_counts;
  return `<header class="hero">
    <h1>BACK<span class="echo">SCATTER</span> — a walk-forward pre-token radar</h1>
    <p class="lede">Protocols that have not launched a token yet, priced by the ones that already trade —
    every price banded by bootstrap, every band proven out of sample, every protocol netted into a
    $1,000 farm plan whose verdict obeys the band's low end, not the point estimate.</p>
  </header>
  <div class="cards">
    <div class="card"><div class="k">Protocols priced</div><div class="v">${R.board_size}</div><div class="s">${R.watchlist_size} watchlist + ${R.early_size} early, below the floor</div></div>
    <div class="card"><div class="k">Traded comps</div><div class="v">${R.model.comps}</div><div class="s">the model's only teachers</div></div>
    <div class="card"><div class="k">Walk-forward RMSLE</div><div class="v">${v.pooled_rmsle}</div><div class="s">naive multiple: ${v.baseline_pooled_rmsle}</div></div>
    <div class="card"><div class="k">Verdicts at $1,000</div><div class="v">${vc.farm} farm · ${vc.thin} thin · ${vc.skip} skip</div><div class="s">judged at the band, not the point</div></div>
    <div class="card"><div class="k">Snapshot</div><div class="v">${esc(R.as_of)}</div><div class="s">floor ${usd(R.floor_usd)}</div></div>
  </div>`;
}

function proofHtml() {
  const v = R.validation;
  const folds = v.folds.map((f) =>
    `<tr><td>${f.fold}</td><td class="num">${f.train_size}</td><td class="num">${f.test_size}</td>` +
    `<td>${esc(f.test_from)} … ${esc(f.test_to)}</td><td class="num">${f.rmsle}</td><td class="num">${f.baseline_rmsle}</td></tr>`).join('');
  const feats = R.model.features.map((f) => `${esc(f.name)} ${f.coef >= 0 ? '+' : ''}${f.coef}`).join(' · ');
  return `<section class="proof"><h2>The proof — walk-forward, the way the market arrives</h2>
  <p>Comps are ordered by the day tracking began; each fold is scored only on comps that listed after
  everything it trained on. The folds chose the ridge penalty (λ = ${v.lambda}) and the blend against the
  naive median multiple (w = ${v.blend_w}) from a stated grid — neither was guessed. Pooled RMSLE
  <strong>${v.pooled_rmsle}</strong> against the naive multiple's <strong>${v.baseline_pooled_rmsle}</strong>;
  fold spread (sd) ${v.fold_rmsle_sd}. A repeated random holdout (${v.holdout_seeds} seeds) averages
  RMSLE ${v.holdout_mean_rmsle} (range ${v.holdout_min_rmsle}–${v.holdout_max_rmsle}). Bands are the
  10th–90th percentile of 400 bootstrap refits, each draw perturbed by the out-of-sample spread
  σ = ${v.sigma_oos} — the bands are wide because the proof says the error is wide.</p>
  <table><thead><tr><th>Fold</th><th class="num">Train</th><th class="num">Test</th><th>Test listed</th><th class="num">RMSLE</th><th class="num">Naive</th></tr></thead>
  <tbody>${folds}</tbody></table>
  <p>Model ${esc(R.model.version)} features: ${feats}. Median comp age ${R.model.median_age_days} days.</p></section>`;
}

function controlsHtml() {
  return `<div class="controls">
    <input type="search" id="q" placeholder="Search name, slug or category…" value="${esc(state.q)}" aria-label="Search the board">
    <select id="verdict" aria-label="Filter by verdict">
      ${['all', 'farm', 'thin', 'skip'].map((x) => `<option value="${x}"${state.verdict === x ? ' selected' : ''}>${x === 'all' ? 'All verdicts' : x.toUpperCase()}</option>`).join('')}
    </select>
    <select id="sort" aria-label="Sort the board">
      ${[['rank', 'Rank (implied value)'], ['tvl', 'TVL'], ['net', 'Net at midpoint'], ['change', '7d change']].map(([x, label]) => `<option value="${x}"${state.sort === x ? ' selected' : ''}>${label}</option>`).join('')}
    </select>
    <label><input type="checkbox" id="watchonly"${state.watchOnly ? ' checked' : ''}> Watchlist only</label>
    <span id="shown"></span>
  </div>`;
}

function visibleRows() {
  let rows = R.rows.filter((r) =>
    (state.verdict === 'all' || r.verdict === state.verdict) &&
    (!state.watchOnly || r.watchlist) &&
    (state.q === '' || (r.name + ' ' + r.slug + ' ' + (r.category ?? '')).toLowerCase().includes(state.q)));
  if (state.sort === 'tvl') rows = [...rows].sort((a, b) => b.tvl_usd - a.tvl_usd);
  else if (state.sort === 'net') rows = [...rows].sort((a, b) => b.net_mid_usd - a.net_mid_usd);
  else if (state.sort === 'change') rows = [...rows].sort((a, b) => (b.change_7d ?? -1e9) - (a.change_7d ?? -1e9));
  return rows;
}

function boardHtml() {
  const rows = visibleRows();
  const body = rows.length === 0
    ? `<tr><td colspan="9"><div class="empty">No protocol matches this view. Loosen the filters.</div></td></tr>`
    : rows.map((r) => `<tr class="row" tabindex="0" data-slug="${esc(r.slug)}">
      <td class="num">${r.rank}</td>
      <td><strong>${esc(r.name)}</strong>${r.watchlist ? ' <span class="chip watch">watchlist</span>' : ''}<br><span class="bandlbl">${esc(r.slug)}</span></td>
      <td>${esc(r.category ?? '—')}</td>
      <td>${esc(r.chains.slice(0, 3).join(', '))}${r.chains.length > 3 ? ' +' + (r.chains.length - 3) : ''}</td>
      <td class="num">${usd(r.tvl_usd)}</td>
      <td class="num">${pct(r.change_7d)}</td>
      <td class="num"><strong>${usd(r.implied_value_usd)}</strong></td>
      <td class="bandcell">${bandHtml(r)}</td>
      <td><span class="chip ${r.verdict}">${r.verdict.toUpperCase()}</span><br><span class="bandlbl ${r.net_mid_usd > 0 ? 'pos' : 'neg'}">net ${usdSigned(r.net_mid_usd)}</span></td>
    </tr>`).join('');
  return `<section class="boardwrap"><h2>The board — pre-token protocols ranked by implied value</h2>
  ${controlsHtml()}
  <table><thead><tr><th class="num">#</th><th>Protocol</th><th>Category</th><th>Chains</th><th class="num">TVL</th><th class="num">7d</th><th class="num">Implied value</th><th>Band (p10–p90)</th><th>Verdict</th></tr></thead>
  <tbody>${body}</tbody></table></section>`;
}

function detailHtml(r) {
  const steps = r.farm_steps.map((s) => `<li>${esc(s)}</li>`).join('');
  return `<section class="drawer" id="detail"><button class="close" id="closex" aria-label="Close detail">Close ×</button>
  <h3>${esc(r.name)} <span class="chip ${r.verdict}">${r.verdict.toUpperCase()}</span>${r.watchlist ? ' <span class="chip watch">watchlist</span>' : ''}</h3>
  <p class="sub">${esc(r.slug)} · ${esc(r.category ?? 'uncategorised')} · ${esc(r.chains.join(', ') || 'no chains recorded')} ·
  tracked since ${esc(r.listed_at ?? 'unknown')}${r.age_days !== null ? ` (${r.age_days} days)` : ''}</p>
  <div class="grid">
    <div class="card"><div class="k">Implied value</div><div class="v">${usd(r.implied_value_usd)}</div><div class="s">band ${usd(r.value_low_usd)} – ${usd(r.value_high_usd)}</div></div>
    <div class="card"><div class="k">TVL</div><div class="v">${usd(r.tvl_usd)}</div><div class="s">7d ${pct(r.change_7d)}</div></div>
    <div class="card"><div class="k">Deposit</div><div class="v">${usd(r.deposit_usd)}</div><div class="s">hold ${r.hold_days} days</div></div>
    <div class="card"><div class="k">Gross (mid / low)</div><div class="v">${usd(r.gross_mid_usd)} / ${usd(r.gross_low_usd)}</div><div class="s">10% allocation, pro-rata</div></div>
    <div class="card"><div class="k">All-in cost</div><div class="v">${usd(r.total_cost_usd)}</div><div class="s">gas + hours at $15 + 4% APY forgone</div></div>
    <div class="card"><div class="k">Net (mid / low)</div><div class="v ${r.net_mid_usd > 0 ? 'pos' : 'neg'}">${usdSigned(r.net_mid_usd)} / ${usdSigned(r.net_low_usd)}</div><div class="s">verdict obeys the low end</div></div>
  </div>
  <div class="bandcell" style="max-width:520px">${bandHtml(r)}</div>
  <h4>The farm plan</h4><ol>${steps}</ol></section>`;
}

function liveHtml() {
  const body = R.live.map((p) => `<tr>
    <td><strong>${esc(p.name)}</strong><br><span class="bandlbl">${esc(p.slug)}</span></td>
    <td>${esc(p.category ?? '—')}</td><td class="num">${usd(p.tvl_usd)}</td>
    <td class="num">${usd(p.mcap_usd)}</td><td class="num">${pct(p.change_7d)}</td>
    <td class="num">${p.model_log_value === null ? '—' : usd(p.model_log_value)}</td></tr>`).join('');
  return `<section class="livewrap"><h2>Already trading — the teachers, at the market's own prices</h2>
  <p>These protocols are never ranked by the model; their market caps are the lessons it learned from.
  The last column is the fitted model's blended value for the same protocol, shown so the fit can be
  held up against reality.</p>
  <table><thead><tr><th>Protocol</th><th>Category</th><th class="num">TVL</th><th class="num">Market cap</th><th class="num">7d</th><th class="num">Model value</th></tr></thead>
  <tbody>${body}</tbody></table></section>`;
}

function footerHtml() {
  return `<footer>
  <p><strong>How to read this board.</strong> A launch is assumed to distribute 10% of implied value to
  depositors pro-rata, so a $1,000 deposit expects implied × 10% × 1000/(TVL + 1000) gross. Gas on a
  stated per-chain schedule, hours at $15/h and 4% APY of forgone yield are charged against it.
  FARM needs the band's low end net positive; THIN (sized at $500) needs the midpoint; the rest SKIP.
  On this snapshot no protocol's low end clears its costs, so the farm count is honestly zero — the
  thin tier is where the midpoint pays.</p>
  <p>Data: ${esc(R.source)}. The run is deterministic: the same snapshot produces this same board,
  byte for byte. No protocol name is special-cased anywhere in the model.</p></footer>`;
}

let detailSlug = null;

function render() {
  const detail = detailSlug ? R.rows.find((r) => r.slug === detailSlug) : null;
  main.innerHTML = headerHtml() + proofHtml() + (detail ? detailHtml(detail) : '') + boardHtml() + liveHtml() + footerHtml();
  main.setAttribute('aria-busy', 'false');
  const shown = document.getElementById('shown');
  if (shown) shown.textContent = `${visibleRows().length} of ${R.rows.length} shown`;
  wire();
  if (detail) document.getElementById('detail').scrollIntoView({ block: 'start' });
}

function wire() {
  const q = document.getElementById('q');
  q.addEventListener('input', () => { state.q = q.value.trim().toLowerCase(); render(); const nq = document.getElementById('q'); nq.focus(); nq.setSelectionRange(nq.value.length, nq.value.length); });
  document.getElementById('verdict').addEventListener('change', (e) => { state.verdict = e.target.value; render(); });
  document.getElementById('sort').addEventListener('change', (e) => { state.sort = e.target.value; render(); });
  document.getElementById('watchonly').addEventListener('change', (e) => { state.watchOnly = e.target.checked; render(); });
  for (const tr of document.querySelectorAll('tr.row')) {
    const open = () => { detailSlug = tr.getAttribute('data-slug'); history.replaceState(null, '', '#' + detailSlug); render(); };
    tr.addEventListener('click', open);
    tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  }
  const close = document.getElementById('closex');
  if (close) close.addEventListener('click', () => { detailSlug = null; history.replaceState(null, '', location.pathname); render(); });
  // A verdict deep link (#thin and friends) lands the reader on the board it filters, not the top.
  const h = decodeURIComponent(location.hash.replace(/^#/, ''));
  if (!detailSlug && ['farm', 'thin', 'skip'].includes(h)) {
    const bw = document.querySelector('.boardwrap');
    if (bw) bw.scrollIntoView({ block: 'start' });
  }
}

function fromHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ''));
  if (['farm', 'thin', 'skip'].includes(h)) { state.verdict = h; detailSlug = null; }
  else if (h && R.rows.some((r) => r.slug === h)) detailSlug = h;
}
fromHash();
window.addEventListener('hashchange', () => { fromHash(); render(); });
render();
