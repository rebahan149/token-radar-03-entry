// BACKSCATTER — pricing protocols that have not issued a token yet, from the ones that have.
//
// The idea in one paragraph: protocols whose tokens already trade are the only teachers a
// pre-token valuation can have, so this model learns from them (the comps), proves itself the
// way the market actually arrives (walk-forward over listing day, never a shuffled split that
// leaks the future into training), bands every price it prints with bootstrap refits perturbed
// by its own demonstrated out-of-sample error, and only then nets each protocol into a farm
// plan sized to a stated $1,000 deposit. A verdict obeys the band, not the point estimate:
// FARM needs the band's low end to clear every cost, THIN needs the midpoint, the rest SKIP.
//
// Everything below reads only the sealed snapshot's own figures. No protocol is special-cased:
// the function that prices a pre-token row is the function that priced the comps in the proof,
// and a fixed seed makes the whole run deterministic — same snapshot in, same board out.

// ------------------------------------------------------------------ snapshot types

/** One protocol in the snapshot: the ten fields the sealed input carries, no more. */
export interface Protocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The snapshot document as the program consumes it. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: Protocol[];
  watchlist: string[];
}

/** Raised when the input is not a snapshot this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

export const MODEL_VERSION = 'backscatter-2';

/** The farm plan's stated constants — one deposit, one allocation share, one wage, one yield. */
export const PLAN = {
  deposit_usd: 1000,
  thin_deposit_usd: 500,
  allocation_share: 0.1,
  wage_usd_per_hour: 15,
  opportunity_apy: 0.04,
} as const;

/** Ridge penalties offered to the walk-forward; it, not taste, picks the winner. */
export const LAMBDA_GRID = [0.3, 1, 3, 10, 30, 100, 300];
/**
 * Blend weights offered alongside: a prediction is w × the ridge's log-value plus (1 − w) × the
 * naive log-value (TVL × the training comps' median cap/TVL multiple). The naive multiple is the
 * ridge's strongest simple rival; leaning on it a little is harder to embarrass out of sample.
 */
export const BLEND_GRID = [1, 0.85, 0.7];
/** Expanding-window folds over the comps, ordered by the day tracking began. */
export const WALK_FOLDS = 5;
/** Bootstrap refits behind every value band. */
export const BOOTSTRAP_DRAWS = 400;
/** The fixed seed: determinism is part of the proof, not an accident of it. */
export const RANDOM_SEED = 20261114;

const DAY_MS = 86_400_000;

// ------------------------------------------------------------------ small numerics

/** mulberry32: a tiny deterministic PRNG, so a rerun reproduces the board byte for byte. */
export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One standard-normal draw from a uniform source (Box–Muller, cosine half). */
function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

const medianOf = (values: number[]): number => {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
};
const meanOf = (values: number[]): number => values.reduce((a, b) => a + b, 0) / values.length;
const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** Solve A x = b by Gaussian elimination with partial pivoting; A and b are left untouched. */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[piv]![col]!)) piv = r;
    const swap = M[col]!; M[col] = M[piv]!; M[piv] = swap;
    const d = M[col]![col]!;
    for (let j = col; j <= n; j++) M[col]![j]! /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r]![col]!;
      if (f === 0) continue;
      for (let j = col; j <= n; j++) M[r]![j]! -= f * M[col]![j]!;
    }
  }
  return M.map((row) => row[n]!);
}

// ------------------------------------------------------------------ snapshot facts

/** Whole days from the day tracking began to the snapshot's day; null when either is unusable. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** A protocol records no token at all when cap, CoinGecko id and symbol are all absent. */
export function isPreToken(p: Protocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;
const EXCHANGE_PREFIX = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/;
/** Categories the snapshot's watchlist rule keeps off a pre-token board: conduits and claims on assets that already trade. */
const CONDUIT_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);

/**
 * Whether a pre-token protocol belongs on the ranked board. The exclusions are the ones the
 * snapshot documents for its own watchlist, applied with fields the snapshot carries: a wrapped,
 * staked or bridged product of a token that already trades (WBTC, Binance staked ETH, USDT0)
 * records no token of its own, but it is not a launch still to come, so it is not ranked.
 */
export function isRankable(p: Protocol): boolean {
  return (
    isPreToken(p) &&
    !CONDUIT_CATEGORIES.has(p.category ?? '') &&
    !VERSIONED.test(p.name) &&
    !WRAPPED_WORD.test(p.name) &&
    !EXCHANGE_PREFIX.test(p.name)
  );
}

/** A teacher: trades today, and carries a TVL, a market cap and a tracking start to order by. */
export function isComp(p: Protocol, asOf: string): boolean {
  return !isPreToken(p) && p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0 && ageDays(asOf, p.listed_at) !== null;
}

// ------------------------------------------------------------------ the ridge fit

/** What a fit must know about the feature space, all of it learned from the comps themselves. */
export interface FeatureSpace {
  /** Feature names in column order; column 0 is the constant. */
  names: string[];
  /** Categories large enough among the comps to earn their own (shrunk) indicator column. */
  categories: string[];
  /** The comps' median age: a protocol with no recorded start is priced as a median-aged one. */
  median_age_days: number;
}

/** The feature space of a comp set: categories with at least five comps, and the median age. */
export function featureSpace(comps: Protocol[], asOf: string): FeatureSpace {
  const counts = new Map<string, number>();
  for (const p of comps) {
    const c = p.category ?? '(none)';
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  const categories = [...counts.entries()].filter(([, n]) => n >= 5).map(([c]) => c).sort();
  const ages = comps.map((p) => ageDays(asOf, p.listed_at)).filter((a): a is number => a !== null);
  return {
    names: ['const', 'log_tvl', 'growth', 'log_age', 'log_chains', ...categories.map((c) => `cat:${c}`)],
    categories,
    median_age_days: ages.length > 0 ? medianOf(ages) : 365,
  };
}

/** One protocol as a feature row in `space`: the only figures the model is allowed to look at. */
export function featureRow(p: Protocol, asOf: string, space: FeatureSpace): number[] {
  const age = ageDays(asOf, p.listed_at) ?? space.median_age_days;
  const growth = clamp(p.change_7d ?? 0, -60, 120) / 100;
  const row = [
    1,
    Math.log(Math.max(1, p.tvl_usd)),
    growth,
    Math.log1p(Math.max(0, age)),
    Math.log(Math.max(1, p.chains.length)),
  ];
  for (const c of space.categories) row.push((p.category ?? '(none)') === c ? 1 : 0);
  return row;
}

/** A fitted ridge: coefficients in standardised space, with the centring that makes them usable. */
export interface RidgeFit {
  coef: number[];
  mean: number[];
  sd: number[];
}

/**
 * Fit log(market cap) on the rows by ridge regression. The continuous features are standardised
 * on the training rows; the constant and the category indicators are centred at zero and never
 * standardised, but every column except the constant carries the penalty.
 */
export function fitRidge(rows: number[][], target: number[], lambda: number): RidgeFit {
  const k = rows[0]!.length;
  const mean = new Array<number>(k).fill(0);
  const sd = new Array<number>(k).fill(1);
  for (let j = 1; j < k; j++) {
    const isIndicator = j >= 5; // columns after log_chains are category indicators
    if (isIndicator) continue;
    const m = meanOf(rows.map((r) => r[j]!));
    const v = meanOf(rows.map((r) => (r[j]! - m) ** 2));
    mean[j] = m;
    sd[j] = Math.sqrt(v) || 1;
  }
  const A = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const b = new Array<number>(k).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const z = rows[i]!.map((v, j) => (v - mean[j]!) / sd[j]!);
    for (let a = 0; a < k; a++) {
      b[a]! += z[a]! * target[i]!;
      for (let c = 0; c < k; c++) A[a]![c]! += z[a]! * z[c]!;
    }
  }
  for (let j = 1; j < k; j++) A[j]![j]! += lambda;
  return { coef: solveLinear(A, b), mean, sd };
}

/** A fitted model's prediction of log(market cap) for one feature row. */
export function predictLog(fit: RidgeFit, row: number[]): number {
  let s = 0;
  for (let j = 0; j < row.length; j++) s += fit.coef[j]! * ((row[j]! - fit.mean[j]!) / fit.sd[j]!);
  return s;
}

const rmsleOf = (pairs: Array<[number, number]>): number =>
  Math.sqrt(meanOf(pairs.map(([pred, actual]) => (pred - actual) ** 2)));

// ------------------------------------------------------------------ the proof

/** One walk-forward fold: what it trained on, what it was allowed to see, how wrong it was. */
export interface FoldResult {
  fold: number;
  train_size: number;
  test_size: number;
  test_from: string;
  test_to: string;
  rmsle: number;
  baseline_rmsle: number;
}

/** The whole validation story, exactly as the dashboard states it. */
export interface Validation {
  comps: number;
  folds: FoldResult[];
  pooled_rmsle: number;
  baseline_pooled_rmsle: number;
  fold_rmsle_sd: number;
  lambda: number;
  blend_w: number;
  lambda_scan: Array<{ lambda: number; w: number; pooled_rmsle: number }>;
  holdout_mean_rmsle: number;
  holdout_min_rmsle: number;
  holdout_max_rmsle: number;
  holdout_seeds: number;
  sigma_oos: number;
}

const byListing = (a: Protocol, b: Protocol): number =>
  a.listed_at! < b.listed_at! ? -1 : a.listed_at! > b.listed_at! ? 1 : a.slug < b.slug ? -1 : 1;

/** The blended log-value: the ridge's prediction leaned against the comps' median multiple. */
function blendedLog(fit: RidgeFit, row: number[], tvlUsd: number, medianMultiple: number, w: number): number {
  return w * predictLog(fit, row) + (1 - w) * Math.log(Math.max(1, tvlUsd) * medianMultiple);
}

/** One expanding-window walk-forward pass at a fixed penalty and blend: folds plus pooled error. */
function walkForward(comps: Protocol[], asOf: string, space: FeatureSpace, lambda: number, w: number): { folds: FoldResult[]; pooled: number; baselinePooled: number } {
  const sorted = [...comps].sort(byListing);
  const n = sorted.length;
  const start = Math.floor(n * 0.45);
  const chunk = Math.max(1, Math.floor((n - start) / WALK_FOLDS));
  const folds: FoldResult[] = [];
  const pooledPairs: Array<[number, number]> = [];
  const pooledBase: Array<[number, number]> = [];
  for (let f = 0; f < WALK_FOLDS; f++) {
    const train = sorted.slice(0, start + f * chunk);
    const test = sorted.slice(start + f * chunk, f === WALK_FOLDS - 1 ? n : start + (f + 1) * chunk);
    if (test.length === 0) continue;
    const fit = fitRidge(train.map((p) => featureRow(p, asOf, space)), train.map((p) => Math.log(p.mcap_usd!)), lambda);
    const medMult = medianOf(train.map((p) => p.mcap_usd! / p.tvl_usd));
    const pairs = test.map((p): [number, number] => [blendedLog(fit, featureRow(p, asOf, space), p.tvl_usd, medMult, w), Math.log(p.mcap_usd!)]);
    const basePairs = test.map((p): [number, number] => [Math.log(p.tvl_usd * medMult), Math.log(p.mcap_usd!)]);
    pooledPairs.push(...pairs);
    pooledBase.push(...basePairs);
    folds.push({
      fold: f + 1,
      train_size: train.length,
      test_size: test.length,
      test_from: test[0]!.listed_at!,
      test_to: test[test.length - 1]!.listed_at!,
      rmsle: round4(rmsleOf(pairs)),
      baseline_rmsle: round4(rmsleOf(basePairs)),
    });
  }
  return { folds, pooled: rmsleOf(pooledPairs), baselinePooled: rmsleOf(pooledBase) };
}

/** Twenty seeded random 80/20 holdouts at one setting: the like-for-like single-holdout yardstick. */
function repeatedHoldout(comps: Protocol[], asOf: string, space: FeatureSpace, lambda: number, w: number): { mean: number; min: number; max: number; seeds: number } {
  const seeds = 20;
  const values: number[] = [];
  for (let s = 1; s <= seeds; s++) {
    const rng = mulberry32(RANDOM_SEED + s * 7919);
    const idx = comps.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = idx[i]!; idx[i] = idx[j]!; idx[j] = t;
    }
    const cut = Math.floor(comps.length * 0.8);
    const train = idx.slice(0, cut).map((i) => comps[i]!);
    const test = idx.slice(cut).map((i) => comps[i]!);
    const fit = fitRidge(train.map((p) => featureRow(p, asOf, space)), train.map((p) => Math.log(p.mcap_usd!)), lambda);
    const medMult = medianOf(train.map((p) => p.mcap_usd! / p.tvl_usd));
    values.push(rmsleOf(test.map((p): [number, number] => [blendedLog(fit, featureRow(p, asOf, space), p.tvl_usd, medMult, w), Math.log(p.mcap_usd!)])));
  }
  return { mean: round4(meanOf(values)), min: round4(Math.min(...values)), max: round4(Math.max(...values)), seeds };
}

/**
 * Validate honestly and let the proof choose the hyperparameters: scan the grid by walk-forward
 * pooled RMSLE, keep the winner, then measure everything — folds, baseline, holdout — at it.
 */
export function validate(comps: Protocol[], asOf: string, space: FeatureSpace): Validation {
  const scan: Array<{ lambda: number; w: number; pooled_rmsle: number }> = [];
  for (const lambda of LAMBDA_GRID) {
    for (const w of BLEND_GRID) {
      scan.push({ lambda, w, pooled_rmsle: round4(walkForward(comps, asOf, space, lambda, w).pooled) });
    }
  }
  const best = scan.reduce((a, b) => (b.pooled_rmsle < a.pooled_rmsle ? b : a));
  const wf = walkForward(comps, asOf, space, best.lambda, best.w);
  const hold = repeatedHoldout(comps, asOf, space, best.lambda, best.w);
  const foldMean = meanOf(wf.folds.map((f) => f.rmsle));
  return {
    comps: comps.length,
    folds: wf.folds,
    pooled_rmsle: round4(wf.pooled),
    baseline_pooled_rmsle: round4(wf.baselinePooled),
    fold_rmsle_sd: round4(Math.sqrt(meanOf(wf.folds.map((f) => (f.rmsle - foldMean) ** 2)))),
    lambda: best.lambda,
    blend_w: best.w,
    lambda_scan: scan,
    holdout_mean_rmsle: hold.mean,
    holdout_min_rmsle: hold.min,
    holdout_max_rmsle: hold.max,
    holdout_seeds: hold.seeds,
    sigma_oos: round4(wf.pooled),
  };
}

// ------------------------------------------------------------------ value bands

/** A protocol's priced future: the point value and the band the proof allows around it. */
export interface Valuation {
  implied_usd: number;
  low_usd: number;
  high_usd: number;
}

/**
 * Price every target. The full-comp fit gives the point estimate; the band comes from bootstrap
 * refits of the comp set, each draw's predictions perturbed by the walk-forward's out-of-sample
 * spread (sigma_oos), so the interval carries both parameter doubt and the model's demonstrated
 * error on comps it had never seen. The band is the 10th–90th percentile of the draws.
 */
export function priceAll(
  comps: Protocol[],
  targets: Protocol[],
  asOf: string,
  space: FeatureSpace,
  lambda: number,
  w: number,
  sigmaOos: number,
): Map<string, Valuation> {
  const rows = comps.map((p) => featureRow(p, asOf, space));
  const target = comps.map((p) => Math.log(p.mcap_usd!));
  const full = fitRidge(rows, target, lambda);
  const fullMult = medianOf(comps.map((p) => p.mcap_usd! / p.tvl_usd));
  const rng = mulberry32(RANDOM_SEED);
  const draws = new Map<string, number[]>(targets.map((p) => [p.slug, []]));
  const targetRows = targets.map((p) => featureRow(p, asOf, space));
  for (let b = 0; b < BOOTSTRAP_DRAWS; b++) {
    const sampleIdx = rows.map(() => Math.floor(rng() * rows.length));
    const fit = fitRidge(sampleIdx.map((i) => rows[i]!), sampleIdx.map((i) => target[i]!), lambda);
    const sampleMult = medianOf(sampleIdx.map((i) => comps[i]!.mcap_usd! / comps[i]!.tvl_usd));
    const noise = gaussian(rng) * sigmaOos;
    for (let t = 0; t < targets.length; t++) {
      draws.get(targets[t]!.slug)!.push(blendedLog(fit, targetRows[t]!, targets[t]!.tvl_usd, sampleMult, w) + noise);
    }
  }
  const out = new Map<string, Valuation>();
  for (let t = 0; t < targets.length; t++) {
    const p = targets[t]!;
    const point = Math.exp(blendedLog(full, targetRows[t]!, p.tvl_usd, fullMult, w));
    const sorted = draws.get(p.slug)!.sort((a, b) => a - b);
    const q = (frac: number): number => Math.exp(sorted[Math.min(sorted.length - 1, Math.floor(frac * sorted.length))]!);
    out.set(p.slug, {
      implied_usd: Math.max(1, Math.round(point)),
      low_usd: Math.max(1, Math.round(q(0.1))),
      high_usd: Math.max(1, Math.round(q(0.9))),
    });
  }
  return out;
}

// ------------------------------------------------------------------ farm plans

export type Verdict = 'farm' | 'thin' | 'skip';

/** A deposit-sized farm plan, fully netted: the verdict and every number that produced it. */
export interface FarmPlan {
  verdict: Verdict;
  deposit_usd: number;
  hold_days: number;
  gross_mid_usd: number;
  gross_low_usd: number;
  gas_usd: number;
  time_cost_usd: number;
  opportunity_cost_usd: number;
  total_cost_usd: number;
  net_mid_usd: number;
  net_low_usd: number;
  steps: string[];
}

const money = (n: number): string => `$${Math.round(n).toLocaleString('en-US')}`;

/** Gas to open a position on a chain, on the plan's stated schedule. */
function setupGas(chain: string): number {
  if (chain === 'Ethereum') return 18;
  if (chain === 'Solana') return 1.5;
  return 2.5;
}
/** Gas for one weekly qualifying action on a chain. */
function weeklyGas(chain: string): number {
  return chain === 'Ethereum' ? 2 : 0.3;
}
/** Gas to claim and exit on a chain. */
function exitGas(chain: string): number {
  return chain === 'Ethereum' ? 6 : 1;
}

/** How long a position must rest to count in more than one snapshot, from the protocol's age. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 90;
  if (age < 365) return 120;
  if (age <= 1095) return 90;
  return 60;
}

/** The category's one concrete opening action, in words a farmer can follow. */
function actionFor(p: Protocol, deposit: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply ${money(deposit)} of USDC to ${name} on ${chain} and leave the position open`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm') || c.includes('farm')) return `Provide ${money(deposit)} of liquidity to ${name}'s main pool on ${chain} and leave the position open`;
  if (c.includes('bridge')) return `Bridge ${money(deposit)} of USDC to ${chain} through ${name} and use the bridged funds once in its app`;
  if (c.includes('staking') || c.includes('restak')) return `Stake ${money(deposit)} of the chain's staking asset through ${name} and keep the receipt token`;
  if (c.includes('derivative') || c.includes('perpetual')) return `Deposit ${money(deposit)} of USDC as margin on ${name} and open one small hedged trade`;
  if (c.includes('prediction')) return `Fund ${money(deposit)} on ${name} and place one position in a liquid market`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit ${money(deposit)} into a ${name} vault on ${chain} and keep the vault shares`;
  if (c.includes('stable') || c.includes('algo')) return `Mint or deposit ${money(deposit)} of the stable asset through ${name} on ${chain} and hold it`;
  if (c.includes('rwa')) return `Mint ${money(deposit)} of the yield asset ${name} offers on ${chain} and hold it`;
  if (c.includes('insurance')) return `Buy ${money(deposit)} of cover from ${name} on ${chain} and keep the policy active`;
  if (c.includes('launchpad')) return `Deposit ${money(deposit)} into ${name}'s current sale or staking pool on ${chain} and keep the receipt`;
  return `Deposit ${money(deposit)} of USDC into ${name} on ${chain} and leave the position open`;
}

/**
 * Net one deposit against a protocol's band and price the work honestly. The verdict is the
 * band's, never the point estimate's: a plan whose low end loses money is never a FARM.
 */
export function farmPlan(p: Protocol, v: Valuation, age: number | null): FarmPlan {
  const chain = p.chains.length > 0 ? p.chains[0]! : 'Ethereum';
  const deposit = PLAN.deposit_usd;
  const hold = holdDaysFor(age);
  const weeks = Math.ceil(hold / 7);
  const share = deposit / (p.tvl_usd + deposit);
  const grossMid = v.implied_usd * PLAN.allocation_share * share;
  const grossLow = v.low_usd * PLAN.allocation_share * share;
  const gas = setupGas(chain) + weeklyGas(chain) * weeks + exitGas(chain);
  const hours = 0.5 + weeks * (10 / 60) + 0.25;
  const timeCost = hours * PLAN.wage_usd_per_hour;
  const opp = deposit * PLAN.opportunity_apy * (hold / 365);
  const totalCost = gas + timeCost + opp;
  const netMid = grossMid - totalCost;
  const netLow = grossLow - totalCost;
  const verdict: Verdict = netLow > 0 ? 'farm' : netMid > 0 ? 'thin' : 'skip';
  const sized = verdict === 'farm' ? deposit : verdict === 'thin' ? PLAN.thin_deposit_usd : deposit;
  const action = actionFor(p, sized, chain);
  const verdictWord = verdict === 'farm' ? 'FARM' : verdict === 'thin' ? 'THIN' : 'SKIP';
  const steps = [
    `Start with a fresh wallet used only for this farm; fund it with ${money(sized)} plus gas money on ${chain} and keep the seed phrase offline. Cost: about ${money(setupGas(chain))} of gas to be ready. Time: 30 minutes.`,
    `${action}. Cost: ${money(sized)} of capital and about ${money(setupGas(chain))} of gas. Time: 15 minutes to set up.`,
    `Make one qualifying action every week for ${weeks} weeks (a deposit, a borrow, a swap or a vote) so the wallet reads as a user, not a drive-by. Cost: about ${money(weeklyGas(chain))} of gas a week. Time: 10 minutes a week.`,
    `Hold the position at least ${hold} days, across more than one snapshot of the protocol's books. Cost: the ${money(sized)} stays in the position and forgoes about ${money(opp)} of safe yield. Time: ${hold} days.`,
    `The arithmetic, before you commit: at the model's ${money(v.implied_usd)} implied value (band ${money(v.low_usd)} to ${money(v.high_usd)}), a ${money(deposit)} pro-rata share of a 10% community allocation expects ${money(grossMid)} gross, ${money(grossLow)} at the band's low end, against ${money(totalCost)} of gas, time and yield given up — net ${money(netMid)}, low end ${money(netLow)}. Verdict: ${verdictWord}. Cost: nothing to read. Time: 5 minutes.`,
    `Claim only from the link on the protocol's own page, never from a search result or a direct message, and never pay a fee to claim. Cost: about ${money(exitGas(chain))} of gas to claim and exit. Time: 15 minutes.`,
    `Log the wallet, the dates and every transaction hash in a farm log, so the position can be repeated, audited or unwound without guesswork. Cost: nothing but the log. Time: 5 minutes.`,
  ];
  return {
    verdict,
    deposit_usd: sized,
    hold_days: hold,
    gross_mid_usd: Math.round(grossMid),
    gross_low_usd: Math.round(grossLow),
    gas_usd: Math.round(gas * 100) / 100,
    time_cost_usd: Math.round(timeCost * 100) / 100,
    opportunity_cost_usd: Math.round(opp * 100) / 100,
    total_cost_usd: Math.round(totalCost * 100) / 100,
    net_mid_usd: Math.round(netMid),
    net_low_usd: Math.round(netLow),
    steps,
  };
}

// ------------------------------------------------------------------ the board

/** One row of board.json: the ranked claim, its band, its verdict and its plan. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  watchlist: boolean;
  age_days: number | null;
  implied_value_usd: number;
  value_low_usd: number;
  value_high_usd: number;
  verdict: Verdict;
  model: { version: string; lambda: number; blend_w: number; sigma_oos: number; comps: number };
  farm_steps: string[];
}

/** The dashboard's view of one board row: everything the page renders, plan included. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  watchlist: boolean;
  implied_value_usd: number;
  value_low_usd: number;
  value_high_usd: number;
  verdict: Verdict;
  deposit_usd: number;
  hold_days: number;
  gross_mid_usd: number;
  gross_low_usd: number;
  total_cost_usd: number;
  net_mid_usd: number;
  net_low_usd: number;
  farm_steps: string[];
}

/** A protocol whose token already trades: shown with the market's own value, never the model's. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
  model_log_value: number | null;
}

/** The fitted model's public face: its features and coefficients, as the dashboard states them. */
export interface ModelCard {
  version: string;
  lambda: number;
  blend_w: number;
  features: Array<{ name: string; coef: number }>;
  comps: number;
  median_age_days: number;
}

/** Everything the dashboard needs: the board, the live set, the model and its proof. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: ModelCard;
  validation: Validation;
  plan_constants: typeof PLAN;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  verdict_counts: Record<Verdict, number>;
  rows: RadarRow[];
  live: LiveRow[];
}

/** How many already-trading protocols the Live view carries, by market cap. */
export const LIVE_ROWS = 24;

/** Build the whole radar from one snapshot: validate, fit, price, plan, rank. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const asOf = snapshot.as_of;
  const comps = snapshot.protocols.filter((p) => isComp(p, asOf));
  const space = featureSpace(comps, asOf);
  const validation = validate(comps, asOf, space);
  const watch = new Set(snapshot.watchlist);
  const targets = snapshot.protocols.filter((p) => isRankable(p) || watch.has(p.slug));
  const values = priceAll(comps, targets, asOf, space, validation.lambda, validation.blend_w, validation.sigma_oos);

  const scored = targets.map((p) => {
    const v = values.get(p.slug)!;
    const age = ageDays(asOf, p.listed_at);
    const plan = farmPlan(p, v, age);
    return { p, v, age, plan };
  });
  scored.sort((a, b) => b.v.implied_usd - a.v.implied_usd || (a.p.slug < b.p.slug ? -1 : 1));

  const modelTag = { version: MODEL_VERSION, lambda: validation.lambda, blend_w: validation.blend_w, sigma_oos: validation.sigma_oos, comps: comps.length };
  const board: BoardRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token',
    watchlist: watch.has(s.p.slug),
    age_days: s.age,
    implied_value_usd: s.v.implied_usd,
    value_low_usd: s.v.low_usd,
    value_high_usd: s.v.high_usd,
    verdict: s.plan.verdict,
    model: modelTag,
    farm_steps: s.plan.steps,
  }));

  const rows: RadarRow[] = board.map((r, i) => {
    const s = scored[i]!;
    return {
      rank: r.rank, slug: r.slug, name: r.name, category: r.category, chains: r.chains,
      tvl_usd: r.tvl_usd, change_7d: r.change_7d, listed_at: r.listed_at, age_days: r.age_days,
      watchlist: r.watchlist, implied_value_usd: r.implied_value_usd,
      value_low_usd: r.value_low_usd, value_high_usd: r.value_high_usd, verdict: r.verdict,
      deposit_usd: s.plan.deposit_usd, hold_days: s.plan.hold_days,
      gross_mid_usd: s.plan.gross_mid_usd, gross_low_usd: s.plan.gross_low_usd,
      total_cost_usd: s.plan.total_cost_usd, net_mid_usd: s.plan.net_mid_usd, net_low_usd: s.plan.net_low_usd,
      farm_steps: r.farm_steps,
    };
  });

  // The live set: protocols that already trade, at the market's own cap. The model's blended
  // value for the same protocol rides along so a reader can hold the fit up against reality.
  const fullRows = comps.map((p) => featureRow(p, asOf, space));
  const fullFit = fitRidge(fullRows, comps.map((p) => Math.log(p.mcap_usd!)), validation.lambda);
  const fullMult = medianOf(comps.map((p) => p.mcap_usd! / p.tvl_usd));
  const liveBySlug = new Map(comps.map((p, i) => [p.slug, Math.round(Math.exp(blendedLog(fullFit, fullRows[i]!, p.tvl_usd, fullMult, validation.blend_w)))]));
  const live: LiveRow[] = snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({
      slug: p.slug, name: p.name, category: p.category, chains: p.chains,
      tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd, change_7d: p.change_7d,
      model_log_value: liveBySlug.get(p.slug) ?? null,
    }));

  const verdictCounts: Record<Verdict, number> = { farm: 0, thin: 0, skip: 0 };
  for (const r of board) verdictCounts[r.verdict]++;

  const radar: RadarData = {
    as_of: asOf,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model: {
      version: MODEL_VERSION,
      lambda: validation.lambda,
      blend_w: validation.blend_w,
      features: space.names.map((name, j) => ({ name, coef: round4(fullFit.coef[j]!) })),
      comps: comps.length,
      median_age_days: space.median_age_days,
    },
    validation,
    plan_constants: PLAN,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: board.filter((r) => !r.watchlist).length,
    live_size: live.length,
    verdict_counts: verdictCounts,
    rows,
    live,
  };
  return { board, radar };
}

// ------------------------------------------------------------------ gate self-check

/** Why a board is not one the gate will take, or null. Mirrors the sealed gate's own rows. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  if (board.length < 100 || board.length > 4000) return `the board holds ${board.length} rows, outside 100..4000`;
  const slugs = new Set(board.map((r) => r.slug));
  if (slugs.size !== board.length) return 'a slug appears twice on the board';
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) return 'every row shares one implied value: not a ranking';
  for (const r of board) {
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0 || r.implied_value_usd > 100_000_000_000_000) return `${r.slug} has implied value ${r.implied_value_usd}`;
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.trim().length < 20 || s.trim().length > 600) return `${r.slug} has a farm step of ${s.trim().length} characters`;
  }
  return null;
}

// ------------------------------------------------------------------ input and output

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
