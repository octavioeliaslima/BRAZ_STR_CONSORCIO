from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 30)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", None)


class STRAnalyzer:
    """
    Short-Term-Rental consórcio cash-flow model.

    This version fixes the alternating “Credit Requested BRL” bug by:
      1.  Storing the bid percentage that actually applies on the
          contemplation month (`embed_at_cont`).
      2.  Using that stored percentage – once – to reduce the credit
          balance for the rest of the term.
    """

    # ------------------------------------------------------------------
    # 0. Constructor
    # ------------------------------------------------------------------
    def __init__(self, params: Dict):
        # Keep an internal copy of params so we can mutate safely
        self.params: Dict = params.copy()
        self._derive_properties()

        # --------------------------------------------------------------
        # Attach bid schedules and random contemplation months
        # --------------------------------------------------------------
        bid_schedules = self.params.get("bid_schedules")
        if not bid_schedules:
            # Build the default alternating schedule once
            term = self.params["term_months"]
            sched = [
                (m, 0.0) if m < 7 else
                (m, 0.5) if 7 <= m <= 12 else
                (m, 0.3) if (m - 13) % 2 == 0 else
                (m, 0.5)
                for m in range(1, term + 1)
            ]
            bid_schedules = [sched for _ in range(len(self.properties))]

        # Make sure we have contemplation probabilities (<= 60 months)
        cont_probs = self.params.get("cont_probabilities")
        if cont_probs is None:
            cont_probs = [1 / 60] * 60  # uniform
        else:
            # Pad / trim to length 60 if necessary
            cont_probs = (cont_probs + [0] * 60)[:60]
            s = sum(cont_probs)
            cont_probs = [p / s for p in cont_probs]
        self.params["cont_probabilities"] = cont_probs

        # Bind schedule, cont_month and the *winning* embed to each prop
        for prop, sched in zip(self.properties, bid_schedules):
            prop["bid_schedule"] = sched
            prop["cont_month"] = int(
                np.random.choice(
                    range(1, min(self.params["term_months"], 60) + 1),
                    p=cont_probs,
                )
            )
            # ---------------------------
            # Store the bid% valid at the contemplation month – this is
            # the only percentage that matters for the remainder.
            # ---------------------------
            prop["embed_at_cont"] = next(
                pct for m, pct in reversed(sched) if m <= prop["cont_month"]
            )
            # Flags
            prop["active_credit"] = True
            prop["active_property"] = False

    # ------------------------------------------------------------------
    # 1. Derive the list of properties from the total credit requested
    # ------------------------------------------------------------------
    def _derive_properties(self) -> None:
        credit_total = self.params["credit_requested_amt"]
        num_cotas = self.params.get("num_cotas", 3)  # default: three 340k letters
        unit_cost = credit_total / num_cotas
        self.properties = [
            {
                "cost": unit_cost,
                "base_daily": unit_cost * self.params["daily_rate_factor"],
                # Flags – will be overwritten later
                "active_credit": True,
                "active_property": False,
            }
            for _ in range(num_cotas)
        ]

    # ------------------------------------------------------------------
    # 2. Build the full monthly projection DataFrame
    # ------------------------------------------------------------------
    def _build_monthly_df(self) -> pd.DataFrame:
        p = self.params
        # Retrieve the parcela redutora percentage
        red_pct = p["parcela_redutora_pct"]

        # Helper to map an index to the current embed according to a schedule.
        def effective_embed(idx: int, schedule: List[Tuple[int, float]]) -> float:
            pct = 0.0
            for month, p_bid in schedule:
                if idx >= month:
                    pct = p_bid
            return pct

        dates = pd.date_range("2025-06-01", periods=p["term_months"], freq="MS")
        start_year = dates[0].year
        rows = []

        credit_total_orig = p["credit_requested_amt"]
        num_cotas = self.params.get("num_cotas", len(self.properties))
        unit_cost = credit_total_orig / num_cotas

        # Running balance helpers for catch-up logic
        missing_brl_per_prop = {i: 0.0 for i in range(len(self.properties))}
        catchup_brl_per_prop = {i: 0.0 for i in range(len(self.properties))}

        # New: cumulative credit granted tracker
        cum_credit_granted = 0.0

        for idx, date in enumerate(dates, 1):
            year_idx = (idx - 1) // 12
            month_idx = (idx - 1) % 12
            factor = (1 + p["adj_install"]) ** year_idx

            # Which properties are already earning STR revenue?
            revenue_props = [prop for prop in self.properties if idx >= prop["cont_month"]]
            credit_props = [prop for prop in self.properties if prop["active_credit"]]

            # ----------------------------------------------------------
            # Consórcio credit balance (indexed by INCC)
            # ----------------------------------------------------------
            unit_indexed_brl = unit_cost * factor

            # The fix: once a quota is contemplated, the balance is reduced
            # by the *stored* embed percentage and stays there forever.
            remains: List[float] = []
            for prop in self.properties:
                if idx < prop["cont_month"]:
                    remains.append(unit_indexed_brl)
                else:
                    remains.append((1 - prop["embed_at_cont"]) * unit_indexed_brl)

            credit_requested_indexed = round(sum(remains))
            credit_full_brl = sum(remains)

            # ----------------------------------------------------------
            # New column: amount of credit granted (financed draw events)
            # ----------------------------------------------------------
            credit_granted_brl = round(
                sum(
                    (1 - prop["embed_at_cont"]) * unit_indexed_brl
                    for prop in self.properties
                    if prop["cont_month"] == idx
                )
            )
            cum_credit_granted += credit_granted_brl

            # ----------------------------------------------------------
            # Consórcio payment components for this month
            # ----------------------------------------------------------
            principal_full_brl = credit_full_brl / p["term_months"]
            admin_brl = (unit_indexed_brl * len(self.properties)   # carta integral indexada
                        * p["consorcio_total_pct"]) / p["term_months"]
            # seguro_pct is annual, so divide original credit by 12 for monthly insurance
            seguro_brl = (credit_total_orig * p["seguro_pct"]) / 12  # pct é anual

            # ----- Missing & catch-up maths per property --------------
            for i, prop in enumerate(self.properties):
                cont_month = prop["cont_month"]
                if idx < cont_month:
                    reduction = (principal_full_brl / len(credit_props) * red_pct if credit_props else 0.0)
                    missing_brl_per_prop[i] += reduction
                elif idx == cont_month:
                    reduction = (principal_full_brl / len(credit_props) * red_pct if credit_props else 0.0)
                    missing_brl_per_prop[i] += reduction
                    remaining = p["term_months"] - idx
                    catchup_brl_per_prop[i] = (missing_brl_per_prop[i] / remaining if remaining else 0.0)
                else:
                    amount = catchup_brl_per_prop[i]
                    missing_brl_per_prop[i] = max(0.0, missing_brl_per_prop[i] - amount)

            # ----- Principal actually due this month ------------------
            # Parcela reduzida (pré-contemplação)  e  catch-up (pós-contemplação)
            principal_per_prop = principal_full_brl / len(self.properties)
            principal_brl = 0.0

            for i, prop in enumerate(self.properties):
                if idx < prop["cont_month"]:
                    # Antes da contemplação: parcela redutora
                    principal_brl += principal_per_prop * (1 - red_pct)

                else:
                    # Depois da contemplação:
                    #   • paga somente a parte que continua financiada
                    #   • + a parcela de recomposição (“catch-up”) daquele imóvel
                    principal_brl += (
                        principal_per_prop * (1 - prop["embed_at_cont"])
                        + catchup_brl_per_prop[i]
                    )

            inst_usd = (principal_brl + admin_brl + seguro_brl) / p["exchange_rate"]

            # ----------------------------------------------------------
            # STR nightly ADR and revenue based on credit granted
            # ----------------------------------------------------------
            if revenue_props:
                avg_daily_brl = sum(
                    prop["base_daily"] * (1 + p["adj_install"]) ** year_idx
                    for prop in revenue_props
                ) / len(revenue_props)
            else:
                avg_daily_brl = 0.0
            daily_rate_usd = avg_daily_brl / p["exchange_rate"] if revenue_props else 0.0

            if not revenue_props:
                rev_usd = admin_usd = 0.0
            else:
                rev_brl = sum(
                    prop["base_daily"] * (1 + p["adj_install"]) ** year_idx
                    * p["monthly_nights"]
                    * p["occupancy_rate"]
                    * p["seasonal_factors"][month_idx]
                    for prop in revenue_props
                )
                rev_usd = rev_brl / p["exchange_rate"]
                admin_usd = rev_usd * p["airbnb_admin_rate"]

            # ----------------------------------------------------------
            # Cash-flow summary
            # ----------------------------------------------------------
            cf_month = rev_usd - (inst_usd + admin_usd)

            # ----------------------------------------------------------
            # Identify the embed bid applied at this month, if any
            # ----------------------------------------------------------
            embed_vals = [
                prop["embed_at_cont"] for prop in self.properties
                if prop["cont_month"] == idx
            ]
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

            # ------------- Flip flags right after recording -------------
            for prop in self.properties:
                if idx == prop["cont_month"]:
                    # property is bought and starts generating revenue,
                    # but it keeps paying its share of the instalment
                    prop["active_property"] = True

        df = pd.DataFrame(rows)

        # Assert that the recomposition balance closed correctly
        assert all(abs(m) < 1 for m in missing_brl_per_prop.values()), "Saldo de recomposição não zerou"

        # ------------------------------------------------------------------
        # Final tidy-up and formatting
        # ------------------------------------------------------------------
        df.rename(columns={
            "Mo. Installment": "Mo. Installment USD",
            "STR Daily Rate": "STR Daily Rate USD",
            "Mo. Gross Revenue": "Mo. Gross Revenue USD",
            "Airbnb Admin Fee": "Airbnb Admin Fee USD",
            "STR Mo. Profit": "STR Mo. Profit USD",
            "Mo. Cash Flow": "Mo. Cash Flow USD",
            "Credit Requested": "Credit Requested BRL",
            # "Credit Granted BRL" stays as-is
        }, inplace=True)

        # Round only the numeric columns; leave the embed column as-is
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].round(0).astype(int)
        return df

    # ------------------------------------------------------------------
    # 3. Aggregate monthly figures into annual summaries
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
        df = df[
            ["Year", "Credit Requested BRL"]
            + [c for c in df.columns if c not in ("Year", "Credit Requested BRL")]
        ]
        df["Annual Flow USD"] = monthly_df.groupby("Year")["Mo. Cash Flow USD"].sum().values

        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].round(0).astype(int)
        return df

    # ------------------------------------------------------------------
    # 4. Public API -----------------------------------------------------
    # ------------------------------------------------------------------
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        monthly_df = self._build_monthly_df()
        annual_df = self._build_annual_df(monthly_df)
        return annual_df, monthly_df


# ----------------------------------------------------------------------
# Quick smoke-test when running the module directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    params = {
        "term_months": 180,  # 15 years
        "adj_install": 0.063,  # annual INCC adjustment
        "exchange_rate": 5.7,  # BRL → USD
        "monthly_nights": 28,
        "occupancy_rate": 0.5,
        "seasonal_factors": [
            1.25, 1.25, 1.20, 1.10, 0.85, 0.65,
            0.70, 0.75, 0.85, 1.10, 1.15, 1.20,
        ],
        "airbnb_admin_rate": 0.25,
        "consorcio_total_pct": 0.25,
        "seguro_pct": 0.00038,
        "credit_requested_amt": 1_020_000,
        "daily_rate_factor": 0.0005,
        "cont_probabilities": [1 / 70] * 70,  # uniform over first 60 months
        "parcela_redutora_pct": 0.374,
    }

    analyzer = STRAnalyzer(params)
    annual_df, monthly_df = analyzer.run()

    print(annual_df)
    print(monthly_df)