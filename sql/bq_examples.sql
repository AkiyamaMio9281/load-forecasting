-- BigQuery examples over the curated series (SPEC section 8, optional).
--
-- Load the cleaned parquet first:
--   bq mk --dataset --location=US energy
--   bq load --source_format=PARQUET energy.pjme_hourly data/processed/pjme_clean.parquet
--
-- Both queries exclude quarantined days (bad_day = 1) for the same reason the backtest
-- does: those hours are interpolated fill, not measurement.

-- 1. Monthly peak load and when it occurred -- the number capacity planning cares about.
WITH hourly AS (
  SELECT
    ts,
    PJME_MW,
    DATETIME(ts, 'America/New_York')       AS local_ts,
    DATE_TRUNC(DATE(ts, 'America/New_York'), MONTH) AS local_month
  FROM `energy.pjme_hourly`
  WHERE bad_day = 0
),
ranked AS (
  SELECT
    local_month,
    PJME_MW,
    local_ts,
    ROW_NUMBER() OVER (PARTITION BY local_month ORDER BY PJME_MW DESC) AS rn
  FROM hourly
)
SELECT
  local_month,
  ROUND(PJME_MW, 0)                        AS peak_mw,
  local_ts                                 AS peak_at_local,
  EXTRACT(HOUR FROM local_ts)              AS peak_hour
FROM ranked
WHERE rn = 1
ORDER BY local_month;

-- 2. Year-on-year change in monthly average load -- weather-driven noise and structural
--    drift both show up here, which is why the forecast baseline is seasonal, not linear.
WITH monthly AS (
  SELECT
    EXTRACT(YEAR  FROM DATETIME(ts, 'America/New_York')) AS yr,
    EXTRACT(MONTH FROM DATETIME(ts, 'America/New_York')) AS mo,
    AVG(PJME_MW) AS avg_mw
  FROM `energy.pjme_hourly`
  WHERE bad_day = 0
  GROUP BY yr, mo
)
SELECT
  yr,
  mo,
  ROUND(avg_mw, 0)                                                AS avg_mw,
  ROUND(LAG(avg_mw) OVER (PARTITION BY mo ORDER BY yr), 0)         AS avg_mw_prev_year,
  ROUND(SAFE_DIVIDE(
    avg_mw - LAG(avg_mw) OVER (PARTITION BY mo ORDER BY yr),
    LAG(avg_mw) OVER (PARTITION BY mo ORDER BY yr)) * 100, 2)      AS yoy_pct
FROM monthly
ORDER BY yr, mo;
