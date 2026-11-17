[![token-radar-01 bounty](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/rebahan149/token-radar-03-entry/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/rebahan149/token-radar-03-entry/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

# Backscatter — a walk-forward pre-token radar

Backscatter prices crypto protocols that have **not launched a token yet** by learning from the
ones that already trade — then it checks its own homework in public, bands every price it prints,
and nets each protocol into a farm plan sized to a concrete **$1,000 deposit**.

## What the current run says (snapshot dated 2026-11-14)

- **758** protocols in the snapshot; **95 traded comps** teach the model; **198** pre-token
  protocols priced and ranked (all 40 watchlist protocols plus 158 early ones below the $50M
  floor; wrapped/staked/bridged products of traded tokens are excluded by the snapshot's own
  documented rule).
- **Walk-forward validation** (expanding window over listing day, 5 folds): pooled **RMSLE
  2.035**, against **2.361** for a naive median-multiple baseline. Per fold: 1.552 / 2.033 /
  1.736 / 1.364 / 2.847.
- **Repeated random holdout** (20 seeds, the single-holdout yardstick averaged so no lucky split
  speaks alone): mean **RMSLE 1.921**, range 1.488–2.363.
- The walk-forward itself chose the ridge penalty (**λ = 300**) and the blend weight (**w = 0.7**
  against the naive multiple) from a stated grid.
- **$1,000 verdicts: 0 farm / 20 thin / 178 skip.** Verdicts are judged at the band's low end, and
  on this snapshot no protocol's low end clears its costs — the farm count is honestly zero; the
  20 thin protocols are where the midpoint pays.
- Top of the board: Steakhouse Financial ($50.3M implied), Sentora Curator ($48.1M), Veda
  ($41.7M), Gauntlet ($40.0M), USDD ($33.3M).

## The model

1. **Comps.** Every snapshot protocol that already trades with a TVL, a market cap and a tracking
   start date. Its lesson is its market cap.
2. **Fit.** Ridge regression of log(market cap) on log(TVL), 7-day TVL growth (clamped), log(age),
   log(chain count) and one shrunk indicator per large category, blended with the naive
   median-multiple value. Features are standardised on the training rows.
3. **Proof.** Walk-forward folds ordered by listing day — each fold is scored only on comps that
   listed after everything it trained on — plus the naive baseline on the same folds, plus a
   20-seed repeated holdout. λ and the blend weight are picked by the walk-forward's pooled RMSLE.
4. **Bands.** Each protocol's value band is the 10th–90th percentile of 400 bootstrap refits of
   the comp set, every draw perturbed by the walk-forward's out-of-sample spread (σ = 2.035 in
   log space). The bands are wide because the proof says the error is wide.
5. **Plans.** A launch is assumed to distribute 10% of implied value to depositors pro-rata, so a
   $1,000 deposit expects `implied × 10% × 1000/(TVL + 1000)` gross. Gas (stated per-chain
   schedule), hours at $15/h and 4% APY of forgone yield are charged against it. **FARM** needs
   the band's low end net positive, **THIN** (sized at $500) needs the midpoint, the rest **SKIP**.

No protocol name is ever special-cased: one function prices every row, and the same function
prices the comps it was tested on. The run is deterministic — the same snapshot produces the
same board, byte for byte.

## Run it

```sh
mkdir -p out
MAKEFUN_OUTPUT_DIR=out node main.ts --input <path-to-snapshot.json>
```

Node 22+, standard library only. The program writes `board.json` and `data.js` to the output
directory, copies `dashboard/` and `screenshots/` beside them, and writes `dossier.json` with the
vendored make.fun library. Open `out/dashboard/index.html` in a browser (with `out/data.js`
beside `out/dashboard/`) — the page renders with no network. Deep links: `#<slug>` opens a
protocol's detail view, `#thin` (or `#farm` / `#skip`) opens the board filtered to that verdict.

## Layout

- `main.ts` — the program: snapshot in, board + dashboard + dossier out.
- `lib/model.ts` — the comps fit, the walk-forward proof, the bootstrap bands, the plan netting.
- `dashboard/` — the page: ranked board with band bars, sortable and filterable, a detail drawer
  per protocol with its netted plan, and a separate view of the protocols that already trade.
- `screenshots/` — the board, a protocol detail view, and the filtered thin tier.
- `vendor/makefun-board.ts` — the make.fun dossier library, as shipped in the competition kit.
- `out/` — the latest run's outputs on the sealed snapshot, left in place by the run itself.
