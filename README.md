# Energy time-series fundamentals

Hands-on Jupyter course on forecasting **electricity load** for the Germany–Luxembourg bidding zone (DE_LU, ENTSO‑E Transparency). Shared helpers live in `[src/utils.py](src/utils.py)` (metrics, parquet load, rolling-origin tooling, plots used in notebook 06).

**Setup.** Python ≥ 3.11. Install deps with `[uv](https://github.com/astral-sh/uv)` (or pip): run `uv sync` in the repo root and select the `.venv` kernel in JupyterLab. For live API pulls (`01_load_data.ipynb`), copy `.env.example` to `.env` and add `ENTSOE_API_KEY`, or export `ENERGY_TS_LOAD_PARQUET` to point notebooks at another parquet path.

---

## Dataset description

- **Variable:** hourly **actual total load** (MW), column `load_mw`.
- **Source:** ENTSO‑E Transparency (via `[entsoe-py](https://github.com/EnergieID/entsoe-py)`); see `01_load_data.ipynb` for document types and bidding-zone EIC.
- **Artifacts:** parquet at `data/raw/de_lu_load_hourly.parquet` (built once from the notebook, then reused everywhere).
- **Preprocessing:** In most notebooks, `[load_hourly_load_series](src/utils.py)` enforces `**Europe/Berlin` timezone**, `**1h`** grid, sorts the index, and **interpolates** missing hours (`limit_direction="both"`) before modeling so series are contiguous for statsmodels workflows.

*(If you regenerated the parquet with different dates than the pinned run below, rerun `06_backtesting.ipynb` and replace the metrics table.)*

---

## Methodology

1. **Exploration & structure** (`01–03`, `02_decomposition`): load, STL / classical decomposition, ACF/PACF.
2. **Baselines & classical models** (`04_baseline_holdout`, `04_ets`, `05_sarima`): simple lag baselines; Holt‑Winters (additive trend & seasonality); SARIMA with `pmdarima` diagnostics.
3. **Comparative backtest** (`06_backtesting`): **rolling‑origin evaluation** (`rolling_origin_eval` in `utils`) with **sliding windows** (60 calendar days × 24 h train per fold unless noted), horizon **24 h**, **8 folds**, step **24 h**.
4. **Models in the leaderboard table:**
  - **Seasonal naive (lag‑24):** $hat{y}*{t+h} = y*{t+h-24}$ (cyclically extended for $h>24$).
  - **ETS:** `statsmodels.tsa.holtwinters.ExponentialSmoothing` — additive trend & seasonality, `seasonal_periods=24`, estimated initialization; **optimized fit per fold**.
  - **SARIMA:** `statsmodels.SARIMAX` with fixed **SARIMA(1,1,1)(1,0,1,24)** (order chosen from prior `auto_arima` work in `05`; not re‑searched every fold).

**Probabilistic add‑on (`06`):** For SARIMA only, Gaussian **95% prediction intervals** from `get_forecast().conf_int(alpha=0.05)` — **coverage** reported as fraction of hourly test draws inside those bounds, averaged across folds.

**Training-time column:** Approximate **wall time to execute the rolling-origin loops** for each model inside `06_backtesting.ipynb` (8 folds, full refit + forecast each fold; SARIMA section includes PI construction). Depends on CPU; rerun with `%%time` or `time.perf_counter()` wrapper on your hardware for exact figures.

---

## Results table — rolling-origin forecast comparison

Point metrics are **means across the eight folds** from `aggregate["mean"]` in `06_backtesting.ipynb`. Values are **rounded** for readability. **Bold** = best (**lowest** MAE–sMAPE, **shortest** training time). **95 % PI coverage** applies **only** to **SARIMA** (Gaussian PIs), so **SARIMA** is in **bold** in that column; point-only models show an em dash.


| Model                             | MAE (MW)  | RMSE (MW) | MAPE (%) | sMAPE (%) | 95% PI coverage† | Training time (indicative)‡ |
| --------------------------------- | --------- | --------- | -------- | --------- | ---------------- | --------------------------- |
| Seasonal naive (lag‑24)           | **2,900** | **3,374** | **5.42** | **5.39**  | —                | **≤ 20 s**                  |
| ETS (Holt‑Winters, add+add, m=24) | 19,327    | 21,625    | 36.80    | 29.10     | —                | 3–15 min                    |
| SARIMA(1,1,1)(1,0,1,24)           | 3,110     | 3,671     | 6.06     | 5.76      | **0.990 (99 %)** | ≈ 5–25 min                  |


† **95% PI coverage**: mean empirical coverage across 8 folds (24 h horizon per fold, SARIMA refit each time via `conf_int(alpha=0.05)`).

‡ **Training time**: order-of-magnitude **wall-clock** to execute that model’s full rolling-origin loop in `06`; depends on CPU — measure locally with `%timeit`/`time.perf_counter()` if needed.

Interpretation on **SARIMA**: ~**99 %** empirical coverage vs a **95 %** nominal PI suggests **Gaussian PIs somewhat wide / conservative** for these folds (paired with PI width summaries in `06`).

---

## Key learnings

- **Same window, apples-to-apples:** A fixed **1440 h sliding** train eliminates confounding train-length effects and keeps SARIMA refits reproducible across folds.
- **Strong naive baseline:** On mean cross-fold scores, **lag‑24 seasonal naive beats SARIMA and massively outscores this ETS spec** — a reminder always to baseline before “big” models.
- **ETS can misbehave badly on short horizons:** Large errors on holidays / regime shifts illustrate **model ≠ automatically robust** despite seasonal structure (`06` fold traces show multimodal errors for ETS).
- **Uncertainty checks matter:** Mean **≈ 99 % empirical 95 % PI** coverage for SARIMA suggests **Gaussian PIs are cautious** relative to nominal 95% across these tiny test slices (read together with PI width outputs in `06`).
- **Shared `utils`** cuts duplicated load/metrics code and keeps notebooks focused on exposition.

---

## Next steps

- **Refit calibration:** sharper PIs via **skew / bootstrap** residuals, or richer **state‑space covariance** tweaks; widen train window sensitivity analysis.
- **ETS:** try **robust/error damped** variants or **automatic model selection**, or constrain parameters when instability appears on short spans.
- **Features & hierarchies:** exogenous regressors (**temperature**, public holidays **DE**), hierarchical **national → sub‑zone**, or probabilistic forecasting with **Temporal Fusion Transformers / LightGBM quantile** pipelines.
- **Packaging:** optional editable install (`pyproject` / setuptools) so notebooks avoid `sys.path.insert` scaffolding.
- **CI / docs:** automate small smoke test on truncated synthetic series matching `statsmodels` API contracts.



"In dispatch and balancing, where would a 5% MAPE day-ahead load forecast actually move money or stability decisions?"

A 5% Mean Absolute Percentage Error (MAPE) in day-ahead load forecasting—while acceptable for some industries—is considered relatively high for large-scale utility operations, where state-of-the-art often falls between 1–3%. 

In dispatch and balancing, this level of error moves money and stability decisions primarily through increased reliance on high-cost, fast-acting balancing markets rather than cheaper day-ahead unit commitment, and by increasing the risk of voltage violations and necessary, expensive redispatching. 

Here is the breakdown from the two perspectives:
1. Electrical Engineering Perspective (Grid Operations & Stability)
The EE focuses on physical constraints (voltage, frequency, thermal limits). A 5% MAPE implies significant "net-load" uncertainty.
Move Stability Decisions (Reserve Management): Operators must hold more spinning reserves (fast-starting gas turbines) to account for that 5% uncertainty. If the load is under-forecasted by 5%, the system risks frequency instability, forcing emergency demand response or rapid generator ramping.
Grid Congestion and Redispatch: A 5% error can cause localized, unexpected congestion on transmission lines. This forces operators to order redispatch—turning down cheap generation and turning up expensive, local generation to avoid overloaded lines, leading to a spike in congestion management costs.
Voltage Violations: Unexpected load fluctuations (the "missing" 5%) can cause voltage levels to deviate from safe operating ranges, requiring activation of automated voltage regulation (AVR) or capacitor banks to prevent outages. 

2. Machine Learning Engineering Perspective (Model Accuracy & Optimization)
The MLE focuses on the "cost" of the prediction itself and how the model behaves at extreme points.
Move Money (Imbalance Charges): A 5% MAPE means the trader/utility is frequently entering the intra-day or real-time balancing market. They will pay a premium to buy back energy (if they over-forecasted) or buy at high spot prices (if they under-forecasted).
Cost of Inaccuracy: While a 1% error reduction can save millions annually, a 5% MAPE implies lower-quality inputs into Unit Commitment models. This leads to inefficient, high-cost, last-minute generation scheduling.
Addressing High MAPE: An MLE would move from a simple model to hybrid models (e.g., LSTM-EMD) because 5% MAPE usually implies that non-linearities (e.g., human behavior, weather impacts) are not being captured. The 5% MAPE highlights a need for feature engineering, such as incorporating smart meter data or more granular weather inputs. 