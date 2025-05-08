"""
STRAnalyzer – Brazilian Consórcio-Financed Short-Term Rental Projection Model
=============================================================================
This module simulates the full life‑cycle cash‑flow of purchasing residential
properties in Brazil by means of a *consórcio de imóveis* (a cooperative credit
pool) and subsequently operating them as short‑term rentals (STR) on platforms
such as Airbnb.  It answers three practical questions for an investor:

1. **How much cash will I have to outlay every month?**  (Consórcio
   installments, insurance, and administrative fees.)
2. **When does each property begin to earn rental income?**  (At the random
   month in which its credit letter is *contemplated* and the winning bid
   percentage is embedded.)
3. **What are the resulting profits and free cash flow, both deterministically
   and under Monte‑Carlo uncertainty in key STR variables?**

The code is organised into five numbered sections that mirror the analytical
pipeline: construction, property derivation, monthly projection, annual
aggregation, and Monte‑Carlo simulation.  Each section begins with a
comprehensive comment block so that line‑by‑line in‑line comments are
unnecessary.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 30)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", None)


class STRAnalyzer:
    """End‑to‑end financial model for a consórcio‑backed STR portfolio.

    The class encapsulates all business logic: credit‑line evolution,
    installment calculations, STR revenue forecasting, cash‑flow assembly, and
    risk analysis by Monte‑Carlo.
    """

    # ------------------------------------------------------------------
    # 0. Constructor – parameter validation & per‑property initialisation
    # ------------------------------------------------------------------
    def __init__(self, params: Dict):
        self.base_params: Dict = params.copy()   # immutable reference for MC runs
        self.params: Dict = params.copy()        # mutable working copy
        self._derive_properties()                # build property‑level stubs

        # ----- Bid schedule creation ----------------------------------
        bid_schedules = self.params.get("bid_schedules")
        if not bid_schedules:
            term = self.params["term_months"]
            default_sched = [
                (m, 0.0) if m < 7 else
                (m, 0.5) if 7 <= m <= 12 else
                (m, 0.3) if (m - 13) % 2 == 0 else
                (m, 0.5)
                for m in range(1, term + 1)
            ]
            bid_schedules = [default_sched for _ in range(len(self.properties))]

        # ----- Contemplation‑month probability vector -----------------
        cont_probs = self.params.get("cont_probabilities")
        if cont_probs is None:
            cont_probs = [1 / 70] * 70  # uniform prior over first 70 months
        else:
            cont_probs = (cont_probs + [0] * 70)[:70]
            s = sum(cont_probs)
            cont_probs = [p_ / s for p_ in cont_probs]

        # zero‑out months where bidding is not allowed, then renormalise
        for m, pct in bid_schedules[0]:
            if m <= len(cont_probs) and pct == 0.0:
                cont_probs[m - 1] = 0.0
        total = sum(cont_probs)
        if total > 0:
            cont_probs = [p_ / total for p_ in cont_probs]
        self.params["cont_probabilities"] = cont_probs

        # ----- Per‑property metadata attachment -----------------------
        for prop, sched in zip(self.properties, bid_schedules):
            prop["bid_schedule"] = sched
            max_month = min(self.params["term_months"], 70)
            valid_months = [m for m, pct in sched if m <= max_month and pct > 0.0]
            probs = [cont_probs[m - 1] for m in valid_months]
            if probs:
                norm_probs = [prob / sum(probs) for prob in probs] if sum(probs) > 0 else []
            prop["cont_month"] = int(np.random.choice(valid_months, p=norm_probs))
            prop["embed_at_cont"] = next(
                pct for m, pct in reversed(sched) if m <= prop["cont_month"]
            )
            prop["active_credit"] = True
            prop["active_property"] = False

    # ------------------------------------------------------------------
    # 1. Property derivation from aggregate credit requested
    # ------------------------------------------------------------------
    # Splits the total credit line into *num_cotas* equal letters and records
    # their notional STR daily rate.  No stochasticity is introduced here so
    # the result is repeatable across deterministic and Monte-Carlo runs.
    # ------------------------------------------------------------------
    def _derive_properties(self) -> None:
        credit_total = self.params["credit_requested_amt"]
        num_cotas = self.params.get("num_cotas", 3)
        unit_cost = credit_total / num_cotas
        self.properties = [
            {
                "cost": unit_cost,
                "base_daily": unit_cost * self.params["daily_rate_factor"],
                "active_credit": True,
                "active_property": False,
            }
            for _ in range(num_cotas)
        ]

    # ------------------------------------------------------------------
    # 2. Monthly projection engine
    # ------------------------------------------------------------------
    # Builds a DataFrame with one row per month over the consórcio term.
    # The loop simultaneously tracks:
    #   • Credit balance evolution (indexed by INCC inflation)
    #   • Consórcio installment breakdown (principal, admin, insurance)
    #   • STR revenue once a property is available to rent
    #   • Free cash flow after fees and installments
    # A running total of *Credit Granted* is maintained for later analysis.
    # ------------------------------------------------------------------
    def _build_monthly_df(self) -> pd.DataFrame:
        p = self.params
        red_pct = p["parcela_redutora_pct"]
        dates = pd.date_range("2025-06-01", periods=p["term_months"], freq="MS")
        start_year = dates[0].year
        rows: List[Dict] = []

        credit_total_orig = p["credit_requested_amt"]
        num_cotas = self.params.get("num_cotas", len(self.properties))
        unit_cost = credit_total_orig / num_cotas

        missing_brl_per_prop = {i: 0.0 for i in range(len(self.properties))}
        catchup_brl_per_prop = {i: 0.0 for i in range(len(self.properties))}
        cum_credit_granted = 0.0

        for idx, date in enumerate(dates, 1):
            year_idx = (idx - 1) // 12
            month_idx = (idx - 1) % 12
            factor = (1 + p["adj_install"]) ** year_idx

            revenue_props = [prop for prop in self.properties if idx >= prop["cont_month"]]
            credit_props = [prop for prop in self.properties if prop["active_credit"]]

            unit_indexed_brl = unit_cost * factor
            remains = [
                unit_indexed_brl if idx < prop["cont_month"] else (1 - prop["embed_at_cont"]) * unit_indexed_brl
                for prop in self.properties
            ]
            credit_requested_indexed = round(sum(remains))
            credit_full_brl = sum(remains)

            credit_granted_brl = round(
                sum(
                    (1 - prop["embed_at_cont"]) * unit_indexed_brl
                    for prop in self.properties
                    if prop["cont_month"] == idx
                )
            )
            cum_credit_granted += credit_granted_brl

            principal_full_brl = credit_full_brl / p["term_months"]
            admin_brl = (unit_indexed_brl * len(self.properties) * p["consorcio_total_pct"]) / p["term_months"]
            seguro_brl = (credit_total_orig * p["seguro_pct"]) / 12

            # ----- Deferred-principal and catch-up logic per property -------
            # Track and repay the “missing” principal that doesn’t get reduced
            # immediately due to the parcela redutora mechanism:
            #   • Pre-contemplation (idx < cont_month):
            #       – Calculate the monthly reduction that *would* have applied
            #         to each active credit line, but defer it by accumulating
            #         into missing_brl_per_prop.
            #   • At contemplation (idx == cont_month):
            #       – Add that month’s deferred amount.
            #       – Compute catchup_brl_per_prop as: total missing BRL ÷ remaining months.
            #         This evenly spreads the backlog across the rest of the term.
            #   • Post-contemplation (idx > cont_month):
            #       – Subtract catchup_brl_per_prop[i] each month from the missing balance
            #         until it’s fully repaid.
            # This ensures that any principal “skipped” early on is systematically
            # reconciled over the remaining term.
            # -------------------------------------------------------------------

            for i, prop in enumerate(self.properties):
                cont_month = prop["cont_month"]
                if idx < cont_month:
                    missing_brl_per_prop[i] += (principal_full_brl / len(credit_props) * red_pct if credit_props else 0.0)
                elif idx == cont_month:
                    missing_brl_per_prop[i] += (principal_full_brl / len(credit_props) * red_pct if credit_props else 0.0)
                    remaining = p["term_months"] - idx
                    catchup_brl_per_prop[i] = (missing_brl_per_prop[i] / remaining if remaining else 0.0)
                else:
                    amount = catchup_brl_per_prop[i]
                    missing_brl_per_prop[i] = max(0.0, missing_brl_per_prop[i] - amount)

            principal_per_prop = principal_full_brl / len(self.properties)
            principal_brl = 0.0
            for i, prop in enumerate(self.properties):
                if idx < prop["cont_month"]:
                    principal_brl += principal_per_prop * (1 - red_pct)
                else:
                    principal_brl += principal_per_prop * (1 - prop["embed_at_cont"]) + catchup_brl_per_prop[i]

            inst_usd = (principal_brl + admin_brl + seguro_brl) / p["exchange_rate"]

            if revenue_props:
                avg_daily_brl = sum(
                    prop["base_daily"] * (1 + p["adj_install"]) ** year_idx for prop in revenue_props
                ) / len(revenue_props)
            else:
                avg_daily_brl = 0.0
            daily_rate_usd = avg_daily_brl / p["exchange_rate"] if revenue_props else 0.0

            if not revenue_props:
                rev_usd = admin_usd = 0.0
            else:
                rev_brl = sum(
                    prop["base_daily"] * (1 + p["adj_install"]) ** year_idx * p["monthly_nights"] * p["occupancy_rate"] * p["seasonal_factors"][month_idx]
                    for prop in revenue_props
                )
                rev_usd = rev_brl / p["exchange_rate"]
                admin_usd = rev_usd * p["airbnb_admin_rate"]

            cf_month = rev_usd - (inst_usd + admin_usd)

            embed_vals = [prop["embed_at_cont"] for prop in self.properties if prop["cont_month"] == idx]
            embed_str = ", ".join(f"{int(v * 100)}%" for v in embed_vals) if embed_vals else 0

            rows.append({
                "Date": date,
                "Year": date.year - start_year + 1,
                "Credit Requested": credit_requested_indexed,
                "Embed At Cont %": embed_str,
                "Credit Granted BRL": cum_credit_granted,
                "Mo. Installment": inst_usd,
                "STR Daily Rate": daily_rate_usd,
                "Mo. Gross Revenue": rev_usd,
                "Airbnb Admin Fee": admin_usd,
                "STR Mo. Profit": rev_usd - admin_usd,
                "Mo. Cash Flow": cf_month,
            })

            for prop in self.properties:
                if idx == prop["cont_month"]:
                    prop["active_property"] = True

        df = pd.DataFrame(rows)
        assert all(abs(m) < 1 for m in missing_brl_per_prop.values()), "Saldo de recomposição não zerou"

        df.rename(columns={
            "Mo. Installment": "Mo. Installment USD",
            "STR Daily Rate": "STR Daily Rate USD",
            "Mo. Gross Revenue": "Mo. Gross Revenue USD",
            "Airbnb Admin Fee": "Airbnb Admin Fee USD",
            "STR Mo. Profit": "STR Mo. Profit USD",
            "Mo. Cash Flow": "Mo. Cash Flow USD",
            "Credit Requested": "Credit Requested BRL",
        }, inplace=True)

        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].round(0).astype(int)
        return df

    # ------------------------------------------------------------------
    # 3. Annual aggregation
    # ------------------------------------------------------------------
    # Collapses the 180-row monthly DataFrame into a year-by-year summary.
    # All monetary columns are averaged except the cash-flow, which is summed
    # to show the annual P&L contribution.
    # ------------------------------------------------------------------
    def _build_annual_df(self, monthly_df: pd.DataFrame) -> pd.DataFrame:
        df = (
            monthly_df.groupby("Year")
            .agg({
                "Mo. Installment USD": "mean",
                "STR Daily Rate USD": "mean",
                "Mo. Gross Revenue USD": "mean",
                "Airbnb Admin Fee USD": "mean",
                "STR Mo. Profit USD": "mean",
                "Mo. Cash Flow USD": "mean",
                "Credit Requested BRL": "first",
            })
            .rename(columns={
                "Mo. Installment USD": "Avg Mo. Installment USD",
                "STR Daily Rate USD": "Avg STR Daily Rate USD",
                "Mo. Gross Revenue USD": "Avg Mo. Gross Revenue USD",
                "Airbnb Admin Fee USD": "Avg Airbnb Admin Fee USD",
                "STR Mo. Profit USD": "Avg STR Mo. Profit USD",
                "Mo. Cash Flow USD": "Avg Mo. Cash Flow USD",
            })
            .reset_index()
        )
        df = df[["Year", "Credit Requested BRL"] + [c for c in df.columns if c not in ("Year", "Credit Requested BRL")]]
        df["Annual Flow USD"] = monthly_df.groupby("Year")["Mo. Cash Flow USD"].sum().values
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].round(0).astype(int)
        return df

    # ------------------------------------------------------------------
    # 4. Public API
    # ------------------------------------------------------------------
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        monthly_df = self._build_monthly_df()
        annual_df = self._build_annual_df(monthly_df)
        return annual_df, monthly_df

    # ------------------------------------------------------------------
    # 5. Monte-Carlo simulation
    # ------------------------------------------------------------------
    # Randomises the STR-specific parameters (occupancy, ADR factor, exchange
    # rate, seasonality, etc.) and re-runs the projection *N* times.
    # Percentiles of both free cash flow and cumulative credit granted are
    # returned as two DataFrames, indexed by Year.
    # ------------------------------------------------------------------
    def run_monte_carlo(self, runs: int = 1000, percentiles: List[int] = [5, 25, 50, 75, 95]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Monte-Carlo wrapper around :py:meth:`run`.

        The probability distributions mirror empirical STR data from major
        Brazilian beach and urban markets and can easily be swapped out or
        parameterised by the caller.
        """
        sim_flows, sim_credit, years = [], [], None
        for _ in range(runs):
            p = self.base_params.copy()
            p["occupancy_rate"] = np.random.beta(7, 4)
            base_daily = self.base_params["daily_rate_factor"]
            p["daily_rate_factor"] = max(0.0, np.random.normal(base_daily, base_daily * 0.12))
            base_ex = self.base_params["exchange_rate"]
            p["exchange_rate"] = max(0.001, np.random.normal(base_ex, base_ex * 0.05))
            p["monthly_nights"] = int(np.random.uniform(24, 28))
            base_seasonal = np.array(self.base_params["seasonal_factors"])
            noise = np.random.uniform(-0.1, 0.1, size=base_seasonal.shape)
            p["seasonal_factors"] = list(np.clip(base_seasonal * (1 + noise), 0.1, None))

            analyzer = STRAnalyzer(p)
            annual_df, monthly_df = analyzer.run()
            sim_flows.append(annual_df["Annual Flow USD"].values)
            credit_by_year = monthly_df.groupby("Year")["Credit Granted BRL"].last().values
            sim_credit.append(credit_by_year)
            if years is None:
                years = annual_df["Year"].values

        flow_matrix = np.vstack(sim_flows)
        credit_matrix = np.vstack(sim_credit)
        pct_flow_df = pd.DataFrame(index=years, columns=[f"P{p}" for p in percentiles])
        pct_credit_df = pd.DataFrame(index=years, columns=[f"P{p}" for p in percentiles])
        for idx, yr in enumerate(years):
            flow_data, credit_data = flow_matrix[:, idx], credit_matrix[:, idx]
            for pctl in percentiles:
                pct_flow_df.loc[yr, f"P{pctl}"] = np.percentile(flow_data, pctl)
                pct_credit_df.loc[yr, f"P{pctl}"] = np.percentile(credit_data, pctl)
        pct_flow_df.index.name = pct_credit_df.index.name = "Year"
        return pct_flow_df.astype(int), pct_credit_df.astype(int)


# ----------------------------------------------------------------------
# Quick smoke test ------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    params = {
        "term_months": 180,
        "adj_install": 0.063,
        "exchange_rate": 5.7,
        "occupancy_rate": 0.55,
        "daily_rate_factor": 0.0006,
        "monthly_nights": 26,
        "seasonal_factors": [
            1.12, 1.10, 1.08, 1.05, 0.93, 0.85,
            0.88, 0.90, 0.95, 1.03, 1.08, 1.13,
        ],
        "airbnb_admin_rate": 0.25,
        "consorcio_total_pct": 0.25,
        "seguro_pct": 0.00038,
        "credit_requested_amt": 1_020_000,
        "cont_probabilities": [1 / 70] * 70,
        "parcela_redutora_pct": 0.374,
    }
    analyzer = STRAnalyzer(params)
    annual_df, monthly_df = analyzer.run()
    print("Deterministic Annual Summary:\n", annual_df)
    print("#" * 200)
    print("Deterministic Monthly Summary:\n", monthly_df)
    print("#" * 200)
    mc_flow_pct, mc_credit_pct = analyzer.run_monte_carlo(runs=1000)
    print("Monte Carlo Percentiles of Annual Flow USD by Year:\n", mc_flow_pct)
    print("#" * 200)
    print("Monte Carlo Percentiles of Cumulative Credit Granted BRL by Year:\n", mc_credit_pct)
