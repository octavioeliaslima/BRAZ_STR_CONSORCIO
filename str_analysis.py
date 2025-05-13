from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 30)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", None)

# New required parameter key:
# "cont_probabilities": List[float] of length up to 60, summing to 1, representing probability distribution for private contemplation month

class STRAnalyzer:
    """
    Short‑Term‑Rental consórcio cash‑flow model wrapped in a reusable class.
    """

    # ------------------------------------------------------------------
    # 0. Constructor
    # ------------------------------------------------------------------
    def __init__(self, params: Dict):
        # keep the same parameter keys that were used in the original
        self.params: Dict = params.copy()
        self._derive_properties()
        # Attach bid schedules per quota
        bid_schedules = self.params.get("bid_schedules")
        if not bid_schedules:
            # Fallback: build the default alternating schedule once
            term = self.params["term_months"]
            sched = [
                (m, 0.0) if m < 7 else
                (m, 0.5) if 7 <= m <= 12 else
                (m, 0.3) if (m - 13) % 2 == 0 else
                (m, 0.5)
                for m in range(1, term + 1)
            ]
            bid_schedules = [sched for _ in range(len(self.properties))]
        for prop, sched in zip(self.properties, bid_schedules):
            prop["bid_schedule"] = sched
            prop["cont_month"] = int(np.random.choice(range(1, min(self.params["term_months"], 60) + 1),
                                                     p=self.params.get("cont_probabilities")))

    # ------------------------------------------------------------------
    # 1. Property list derived from credit amount
    # ------------------------------------------------------------------
    def _derive_properties(self) -> None:
        credit_total = self.params["credit_requested_amt"]
        num_cotas = self.params.get("num_cotas", 3)  # default to three R$340k cotas
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
    # 2. Build the full monthly projection DataFrame
    # ------------------------------------------------------------------
    def _build_monthly_df(self) -> pd.DataFrame:
        p = self.params
        # --- Bid / embed parameters -----------------------------------
        # Map schedule to an effective embed % for each month index
        from typing import List, Tuple
        def effective_embed(idx: int, schedule: List[Tuple[int, float]]) -> float:
            """
            Return the bid percentage active for *idx* based on this quota's schedule.
            Uses the last (month, pct) with month <= idx.
            """
            pct = 0.0
            for month, p_bid in schedule:
                if idx >= month:
                    pct = p_bid
            return pct
        dates = pd.date_range("2025-06-01", periods=p["term_months"], freq="MS")
        start_year = dates[0].year
        rows = []
        
        credit_requested_orig = p["credit_requested_amt"]  # face value grows with INCC
        # cost per individual quota (BRL)
        num_cotas = self.params.get("num_cotas", len(self.properties))
        unit_cost = credit_requested_orig / num_cotas

        # running balance helpers per property
        missing_brl_per_prop = {i: 0.0 for i in range(len(self.properties))}
        catchup_brl_per_prop = {i: 0.0 for i in range(len(self.properties))}

        for idx, date in enumerate(dates, 1):

            year_idx = (idx - 1) // 12
            month_idx = (idx - 1) % 12

            factor = (1 + p["adj_install"]) ** year_idx

            # Determine which properties have started STR revenue
            revenue_props = [prop for prop in self.properties if idx >= prop["cont_month"]]
            credit_props = [prop for prop in self.properties if prop["active_credit"]]

            # -- Consórcio payment components --
            credit_factor = factor
            unit_indexed_brl = unit_cost * credit_factor

            # Adjust requested credit by each property's embedded bid percentage
            remains = []
            for prop in self.properties:
                embed_pct = effective_embed(idx, prop["bid_schedule"])
                remains.append((1 - embed_pct) * unit_indexed_brl)
            credit_requested_indexed = round(sum(remains))
            credit_full_brl = sum(remains)

            # Consórcio payment components for this month (based on full indexed credit)
            principal_full_brl = credit_full_brl / p["term_months"]
            admin_brl = (credit_full_brl * p["consorcio_total_pct"]) / p["term_months"]
            seguro_brl = credit_full_brl * p["seguro_pct"]

            # Calculate missing and catchup sums per property for this month
            total_reduction = 0.0
            total_catchup = 0.0
            for i, prop in enumerate(self.properties):
                cont_month = prop["cont_month"]
                if idx < cont_month:
                    reduction = principal_full_brl / len(credit_props) * 0.374 if len(credit_props) > 0 else 0.0
                    missing_brl_per_prop[i] += reduction
                elif idx == cont_month:
                    reduction = principal_full_brl / len(credit_props) * 0.374 if len(credit_props) > 0 else 0.0
                    missing_brl_per_prop[i] += reduction
                    remaining = p["term_months"] - idx
                    catchup_brl_per_prop[i] = missing_brl_per_prop[i] / remaining if remaining else 0.0
                else:
                    # After contemplation, reduce missing balance by the catch-up amount
                    amount = catchup_brl_per_prop[i]
                    missing_brl_per_prop[i] = max(0.0, missing_brl_per_prop[i] - amount)

            # Sum total missing and catchup across all properties active for credit
            total_missing = sum(missing_brl_per_prop[i] for i, prop in enumerate(self.properties) if prop["active_credit"])
            total_catchup = sum(catchup_brl_per_prop[i] for i, prop in enumerate(self.properties) if prop["active_credit"])

            # Calculate principal_brl with catchup and missing adjustment
            if len(credit_props) == 0:
                principal_brl = 0.0
            else:
                # Sum principal for all active credit quotas
                principal_brl = 0.0
                for i, prop in enumerate(self.properties):
                    if prop["active_credit"]:
                        cont_month = prop["cont_month"]
                        if idx < cont_month:
                            principal_brl += principal_full_brl / len(credit_props) - 0.374 * principal_full_brl / len(credit_props)
                        elif idx == cont_month:
                            principal_brl += principal_full_brl / len(credit_props) - 0.374 * principal_full_brl / len(credit_props)
                        else:
                            principal_brl += principal_full_brl / len(credit_props) + catchup_brl_per_prop[i]

            inst_usd = (principal_brl + admin_brl + seguro_brl) / p["exchange_rate"]

            # -- STR nightly ADR in USD --
            if revenue_props:
                avg_daily_brl = sum(
                    prop["base_daily"] * (1 + p["adj_install"]) ** year_idx
                    for prop in revenue_props
                ) / len(revenue_props)
            else:
                avg_daily_brl = 0.0
            daily_rate_usd = avg_daily_brl / p["exchange_rate"]
            # Hide ADR until after contemplation for each prop, so if no revenue_props, zero
            if len(revenue_props) == 0:
                daily_rate_usd = 0

            # -- Revenue & fees (Airbnb Admin fee includes host/platform fee) --
            if len(revenue_props) == 0:
                rev_usd = admin_usd = 0
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

            # -- Cash flow calculations --
            cf_month = rev_usd - (inst_usd + admin_usd)

            rows.append(
                {
                    "Date": date,
                    "Year": date.year - start_year + 1,
                    "Credit Requested": credit_requested_indexed,
                    "Mo. Installment": inst_usd,
                    "STR Daily Rate": daily_rate_usd,
                    "Mo. Gross Revenue": rev_usd,
                    "Airbnb Admin Fee": admin_usd,
                    "STR Mo. Profit": rev_usd - admin_usd,
                    "Mo. Cash Flow": cf_month,
                }
            )

            # Deactivate any property that contemplates this month
            for prop in self.properties:
                if prop["active_credit"] and idx == prop["cont_month"]:
                    prop["active_credit"] = False
                    prop["active_property"] = True

        df = pd.DataFrame(rows)
        assert all(abs(missing) < 1 for missing in missing_brl_per_prop.values()), "Saldo de recomposição não zerou"
        # Suffix " USD" to currency column names
        df.rename(columns={
            "Mo. Installment":   "Mo. Installment USD",
            "STR Daily Rate":    "STR Daily Rate USD",
            "Mo. Gross Revenue": "Mo. Gross Revenue USD",
            "Airbnb Admin Fee":  "Airbnb Admin Fee USD",
            "STR Mo. Profit":    "STR Mo. Profit USD",
            "Mo. Cash Flow":     "Mo. Cash Flow USD",
            "Credit Requested":  "Credit Requested BRL",
        }, inplace=True)

        # Round all numeric columns to 0 decimals for cleaner display
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].round(0).astype(int)

        return df

    # ------------------------------------------------------------------
    # 3. Aggregate monthly DF into annual metrics
    # ------------------------------------------------------------------
    def _build_annual_df(self, monthly_df: pd.DataFrame) -> pd.DataFrame:
        df = (
            monthly_df.groupby("Year")
            .agg({
                "Mo. Installment USD":      "mean",
                "STR Daily Rate USD":       "mean",
                "Mo. Gross Revenue USD":    "mean",
                "Airbnb Admin Fee USD":     "mean",
                "STR Mo. Profit USD":       "mean",
                "Mo. Cash Flow USD":        "mean",
                "Credit Requested BRL":     "first",
            })
            .rename(columns={
                "Mo. Installment USD":   "Avg Mo. Installment USD",
                "STR Daily Rate USD":    "Avg STR Daily Rate USD",
                "Mo. Gross Revenue USD": "Avg Mo. Gross Revenue USD",
                "Airbnb Admin Fee USD":  "Avg Airbnb Admin Fee USD",
                "STR Mo. Profit USD":    "Avg STR Mo. Profit USD",
                "Mo. Cash Flow USD":     "Avg Mo. Cash Flow USD",
            })
            .reset_index()
        )
        # Suffix " BRL" to credit column
        df.rename(columns={"Credit Requested": "Credit Requested BRL"}, inplace=True)
        df = df[["Year", "Credit Requested BRL"] + [col for col in df.columns if col not in ("Year", "Credit Requested BRL", "Platform Fee")]]
        df["Annual Flow USD"] = monthly_df.groupby("Year")["Mo. Cash Flow USD"].sum().values
        # Integer formatting like the original script
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


# ----------------------------------------------------------------------
# Quick smoke‑test when running the module directly
# ----------------------------------------------------------------------
if __name__ == "__main__":

    params = {
        "embed_pct": 0.3,
        "term_months": 180,                 # total duration of the consórcio in months
        "adj_install": 0.063,               # annual % increase for installments & nightly ADR (INCC)
        "exchange_rate": 5.7,               # BRL → USD conversion rate used throughout
        "monthly_nights": 29.5,             # nights per month the unit is available for STR
        "occupancy_rate": 0.60,             # ~50 % Florianópolis avg + small premium for Campeche   # 19.2 d
        "seasonal_factors": [               # Jan→Dec demand multipliers
            1.25,  # Jan: strong summer
            1.25,  # Feb: strong summer
            1.20,  # Mar
            1.10,  # Apr: shoulder season
            0.85,  # May: early low season
            0.65,  # Jun: mid‑year trough 
            0.70,  # Jul: slight rebound over June
            0.75,  # Aug
            0.85,  # Sep: late shoulder
            1.10,  # Oct: pre‑summer build
            1.15,  # Nov: summer ramp
            1.20,  # Dec: full summer
        ],
        "platform_rate": 0.00,              # platform fee (e.g., Whimstay/Airbnb) as % of gross
        "airbnb_admin_rate": 0.25,          # Airbnb admin + cleaning/tax bundle as % of gross
        "consorcio_total_pct": 0.25,        # total consórcio admin cost (25 % of credit letter)
        "seguro_pct": 0.00038,              # monthly insurance on balance (0.038 % of credit)
        "credit_requested_amt": 1_020_000,  # credit letter value requested (BRL)
        "daily_rate_factor": 0.00065,       # factor to derive nightly ADR from property price
        "cont_probabilities": [1/60]*60     # uniform distribution over first 60 months
    }

    annual_df, monthly_df = STRAnalyzer(params).run()
    
    
    print(annual_df)
    print('#################################################################################################################################################################')
    # print(monthly_df.head(20))
    # print(monthly_df.tail(20))
    print('#################################################################################################################################################################')
    print(monthly_df)