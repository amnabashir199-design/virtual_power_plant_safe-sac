# Data Provenance

## Benchmark status

The publication dataset is a hybrid benchmark. It is not a measured single-site data-center VPP. Source timelines were aligned by construction, and several variables were modeled, derived, scaled, or generated.

## Source-based layers

| Layer | Source identifier | Use | Redistribution treatment |
|---|---|---|---|
| IT demand and utilization | NLR-labeled GenAI whole-facility archive; internal file `03_whole-facility_profiles/inference/simulated_data/inference_1MW_283nodes_40u_power.csv` | One-minute 2018 inference profile shifted to 2019 and resampled to 15 minutes | Raw archive excluded; authors must provide stable source identifier and confirm terms |
| German grid/market/renewables | Open Power System Data, Time series, version 2020-10-06, `https://doi.org/10.25832/time_series/2020-10-06` | Load, load forecast, wind/solar generation, solar profile, and day-ahead price for 2019 | Raw OPSD file excluded; cite OPSD and its upstream sources |
| Cooling/thermal context | `cold_source_control_dataset.csv`; candidate upstream location `https://gitlab.univ-nantes.fr/E25C590R/datacenter/-/blob/main/cold_source_control_dataset.csv` | Ambient/inlet/outlet and cooling-pattern context tiled over 2019 | Raw file excluded; authors must confirm exact upstream version and terms |

Static background datasets and auxiliary weather/dispatch/battery archives were not merged into the row-level publication benchmark and are excluded.

## Modeled and derived layers

- Aggregate cooling demand applies the aligned cooling ratio to IT demand.
- Auxiliary demand is 5% of IT demand.
- Local PV uses the German OPSD solar profile scaled to 1,000 kWp.
- Export price is 50% of the nonnegative import price.
- Renewable share is `clip((grid solar + grid wind) / grid load, 0, 1)`.
- Carbon proxy is `clip(650 - 500 × renewable share, 150, 650)` g CO2-proxy/kWh. It is not metered or consequential emissions.
- Forecast observations use causal, one-step-lagged, four-interval rolling means with training-median fallback.
- The effective grid contract limit is the training-only 90th percentile of no-battery import: 676.4698656123257 kW.
- The VPP reference uses the causal one-step-lagged trailing 96-interval mean of no-battery import, training-median fallback, and clipping to `[0, 676.4698656123257]` kW; it is additionally capped by the DR limit during events.
- DR stress days satisfy high-price **or** high grid-load-forecast conditions using training 85th-percentile thresholds (63.67 EUR/MWh and 70,536 MW), restricted to Tuesday/Thursday/Friday. Active events are 16:00–18:00 UTC.
- The DR import limit is forecast import minus `min(0.35 × forecast import, 450 kW)`, lower-bounded at zero.

## Integrity

The publication Parquet contains 35,040 rows spanning 2019-01-01 00:00 through 2019-12-31 23:45 UTC without duplicate or missing timestamps. Its expected SHA256 is recorded in `data/metadata/dataset_metadata.json` and the package manifest.

## Public-release caution

The processed benchmark is included for exact evaluation reproducibility. Before repository deposition, the authors must confirm whether the transformations and selected columns can be redistributed under all applicable source terms. This package makes no claim that the raw sources or processed benchmark are already public.
