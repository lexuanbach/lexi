# Real datasets used by the real-trace grounding (RQ2-real)

These files ground the simulator's carbon-intensity and workload signals in real,
publicly available traces used by peer work, so the primary comparison can be
re-run under real conditions (see `real_data_experiment.py`).

## `real_carbon_uk.json` — grid carbon intensity (gCO2eq/kWh)
- **Source:** UK National Grid ESO *Carbon Intensity API*
  (https://carbonintensity.org.uk), regional endpoint, fully open, no auth.
- **Content:** 14 days (2024-01-01 to 2024-01-15) of half-hourly *forecast*
  carbon intensity for three contrasting regions, mapped to the simulator's three
  regions to give real spatial + temporal variation:
  - `NorthScotland` (region 0) — clean grid, mean 92.8, range 0-338
  - `London`        (region 1) — mid,   mean 167.5, range 31-365
  - `Yorkshire`     (region 2) — dirtier, mean 182.9, range 51-276
- Downloaded via the regional `/regional/intensity/{from}/{to}` endpoint.

Electricity Maps (electricitymaps.com) publishes the same quantity per zone; its
historical CSVs require a (free) portal login, so we use the equivalent open UK
API for a fully reproducible artifact.

## `real_arrivals_azure.json` — serverless invocation arrivals
- **Source:** Microsoft *Azure Functions Trace 2019*
  (Shahrad et al., USENIX ATC 2020; https://github.com/Azure/AzurePublicDataset),
  `invocations_per_function_md.anon.d01.csv`.
- **Content:** total invocations per minute (`per_min`, 1440 values) and per hour
  (`hourly`, 24 values), aggregated over all 46,412 functions of day 1
  (909,783,379 invocations). Drives the simulator's request-load time series
  (real diurnal shape and burstiness), normalised to the simulator's load range.

## Service-DAG topology
The simulator's request DAG (frontend -> auth -> {search, recommend} -> order -> pay,
with auth/pay PII-carrying) follows the shape of DeathStarBench
(Gan et al., ASPLOS 2019) and Alibaba microservice traces (Luo et al., SoCC 2021).
