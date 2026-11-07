# Breakpoint 4: paper execution, reflections and dynamic validation

The policy now has a paper execution boundary with durable intent and research
journals, broker reconciliation, pending-order reservations, and independent
risk checks. The model has no order-placement tool. Broker transport is fixed to
`https://paper-api.alpaca.markets`; redirects and automatic POST retries are
blocked. No broker requests, model training or production inference were run
while implementing this breakpoint.

## Execution choices

Orders are whole-share, long-only **IOC limit orders**, with extended hours
explicitly disabled. A buy uses the entitled ask plus the configured buffer,
rounded up to cents; a sell uses the bid minus the buffer, rounded down. The risk
check uses that final price once. IOC cancels an unfilled remainder rather than
leaving an old signal queued overnight. Partial fills and cancellations remain
broker outcomes, not assumed complete fills. The runner stops near market close.
See Alpaca's [order documentation](https://docs.alpaca.markets/us/docs/orders-at-alpaca)
for IOC semantics and [latest quotes](https://docs.alpaca.markets/us/reference/stocklatestquotes-1)
for the entitled IEX data used by the independent execution checks.

Each POST has a stable client ID derived from run, cycle and symbol. The complete
immutable payload and submitting state commit before POST. A lost response,
server failure or restart during submission triggers lookup by client order ID;
404 does not authorize another POST. Unknown state pauses execution. The outbox
binds account, paper endpoint, run, code/model/source identity and risk config.
It verifies intent checksums and cumulative fill deltas before using them.

Broker cash, positions and equity remain authoritative. Pending buys reserve
cash and exposure; pending sales reserve shares and do not fund new buys. Risk
caps use actual broker equity, not hypothetical gains on unfilled orders. A
conservative local confirmed-fill cash budget protects reservations, and cash or
position mismatches pause until broker state reconciles. Unexplained splits,
manual trades or account resets require review before FIFO lessons can proceed. Partial-fill notional is calculated from cumulative quantity times
average price, subtracting previously recorded cumulative notional. Position
updates must reconcile before the fill is released for trading or reflection.
External orders, unsupported/fractional/short holdings, unexpected order states,
missing quotes and incomplete open-order snapshots pause the runner.

## Sources and memory

Archive prices and language records seed the full allowed universe. Before the
policy cutoff is fixed, a bounded live refresh selects held stocks first, then
absolute five-session raw-close movers, up to eight stocks. It refreshes six
months of completed raw prices, the latest news page and recent SEC filings.
Historical preparation still uses complete news pagination and historical SEC
shards; these live-only bounds do not truncate the training archive.

The policy receives the later immutable cutoff, actual broker cash/holdings,
source-refresh status, eligible raw news/SEC text and eligible BM25 memories.
Live price receipt times cannot be backdated into earlier snapshots. Archive
and live price bars with different adjustment policies are not mixed. Fresh
execution quotes are fetched separately from any daily Alpha Vantage research
quote. The same shared free Alpha Vantage quota and persistent cache still apply.

Delayed reflections use only reconciled, matched own BUY/SELL lots. Initial
holdings with unknown cost basis are consumed without inventing a realized PnL.
Confirmation-delta IDs and receipt times are explicitly distinguished from
exchange execution IDs. Paper fees are assumed zero and no dividends are
attributed, matching the documented limitations of Alpaca's
[paper simulator](https://docs.alpaca.markets/us/v1.4.2/docs/paper-trading).
The Qwen response must echo the authoritative stock, action, outcome and PnL.
Its lesson remains an uncertain observation, not proof of a causal trading edge.

Generation attempts are durable. An uncertain or failed reflection generation
is not automatically repeated until it gives a preferred lesson. A saved valid
lesson can retry only its idempotent memory write. Memory is persisted and
isolated by experiment/run/fold; only lessons available by the next decision
cutoff can be retrieved. Retrieval actually feeds the bounded policy prompt.

## Review and commands

Use a dedicated paper account with no external open orders. Keep its outbox and
memory on durable **local** storage. Do not delete or replace a pending outbox to
bypass a reconciliation pause. Back up the closed SQLite database, or use SQLite
backup, rather than copying a database during an active transaction.

`configs/paper.example.json` preserves all 100 assignment symbols and the replay
risk defaults. Its sector map deliberately groups everything as `unknown`, so the
30% sector cap conservatively covers the whole universe, matching the current
replay assumption. If you choose real sectors or different thresholds, freeze
and validate those choices together; a changed config is a new experiment.

The existing credentials are sufficient: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
`SEC_USER_AGENT`, and optionally `ALPHA_VANTAGE_API_KEY` and `HF_TOKEN`. A credential
file is parsed as assignments, not executed as shell code; quote values containing
spaces. Exported environment values take precedence. Keys are never printed.

```bash
cd ~/llm_trading_agent
# Replace these with the complete evidence archive and chosen immutable adapter.
PREPARED="$HOME/llm_trading_agent/artifacts/complete-news-sec"
ADAPTER="$PREPARED/sft"
STATE="$HOME/stock-paper-state"
mkdir -p "$STATE"

# Read account state/reconcile. No generation or order POST.
.venv/bin/python -m agent.paper_cli status \
  --config configs/paper.example.json --prepared "$PREPARED" --adapter "$ADAPTER" \
  --outbox "$STATE/outbox.sqlite" --memory "$STATE/memory.sqlite" \
  --run-id paper-v1 --env-file .env

# During an open regular session: research and risk checks, no order POST.
.venv/bin/python -m agent.paper_cli cycle \
  --config configs/paper.example.json --prepared "$PREPARED" --adapter "$ADAPTER" \
  --outbox "$STATE/outbox.sqlite" --memory "$STATE/memory.sqlite" \
  --run-id paper-v1 --cycle-id dry-2026-11-09-1500Z --env-file .env
```

After reviewing the actual proposal and completing candidate validation, use a
new stable cycle ID and add `--submit-paper` to the cycle command to place paper
orders. Repeating that cycle ID reconciles/reuses its saved proposal; it does not
regenerate until it gets a trade or place another copy. A dry-run ID cannot be
promoted to a submitting cycle. Model signals expire after 300 seconds; execution
checks quote/snapshot freshness again immediately before POST.

A crash before submitting state can leave an unsent intent. Inspect the ledger,
then `discard-unsent` with the same configuration/paths/run arguments and
`--client-id qwen-...` can discard only an intent that never entered submission.
It cannot discard unknown/submitting orders. Create a fresh cycle after that
repair; expired proposals are not silently resubmitted. Configuration, source,
adapter or code changes require a separate declared run, not bypassing a pause.

See [research-evaluation.md](research-evaluation.md) for CPU planning and the
resumable two-session → complete validation job. That CLI intentionally disables
memory for an identical empty-seed base/SFT/PPO comparison and reports the
choice. It does not yet establish the performance of the memory-enabled paper
loop. The current v5 archive still needs raw news for NLP trajectory training and
historical validation; aggregated news cannot reconstruct missing article text.

## Remaining release gates

This breakpoint implements the broker/memory/dynamic-validation foundation.
Production paper dry-run and fault observations, GPU candidate comparisons,
final notebook-facing submission entrypoints, immutable release packaging, and
the held-out execution schedule remain explicit release work. Nothing here
claims a trained candidate is valid or profitable.

The October week must end at **October 30, 2026 close**. Close-to-next-open
execution from October 26–30 decisions would reach November 2. A full five-session
October 26–30 test needs Friday October 23 evidence for Monday opening execution,
then Monday–Thursday closing decisions. Alternatively disclose a Monday cash-only
day and four opportunities. The validation command exposes no test switch; do
not change split boundaries to obtain a test score before freezing that protocol.


The offline demo remains supported. Its legacy broker stub points to this paper
entrypoint; it does not submit orders outside the new executor. Set the paper
account to the assignment's $100,000 starting balance before the assessed run and
archive its first broker snapshot. The status command reports the actual account;
it does not reset a balance or liquidate pre-existing holdings.
