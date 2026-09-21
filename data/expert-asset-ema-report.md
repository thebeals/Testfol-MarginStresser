# Individual-Asset EMA Protection Report

Signals use each asset's prior-day close versus its EMA. Changes apply on the next session. Individual mode removes under-EMA assets and redistributes to remaining active targets; full_cash mode exits everything when the below-EMA count reaches the threshold.

Tested **120** configurations across the expert top 10.

| Rank | Window | Below EMA Trigger | Mode | CAGR Change | DD Improvement |
|---:|---:|---:|---|---:|---:|

## Interpretation

Individual mode preserves assets that remain above the EMA while moving weak sleeves to cash. Full-cash mode is more defensive but can discard good holdings because of one or two weak components. Re-entry can lag sharp rebounds. These results are historical and require separate out-of-sample validation.
