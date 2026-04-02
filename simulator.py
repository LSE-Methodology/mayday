"""
A/B Test Simulator for the Predict-Ship-Compete workshop.

Generates synthetic live traffic and routes it through each team's model.
Each team gets the same users/ads but their model decides which ad to show.
The DGP determines click/convert outcomes. Revenue is tracked per team.

Latency penalty: if a model's inference exceeds the budget, a fraction of
its traffic gets a random (unoptimized) ad choice instead — naturally
punishing slow models without completely zeroing them out.
"""

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class ABTestSimulator:
    def __init__(self, db_path: Path, teams: dict[str, dict[str, Any]]):
        self.db_path = db_path
        self.teams = teams
        self.running = False
        self.total_requests = 0
        self.requests_per_second = 20
        self.latency_budget_ms = 50
        self.rng = np.random.default_rng(12345)

        # Load reference data
        conn = sqlite3.connect(db_path)
        self.users_full = pd.read_sql("SELECT * FROM _users_full", conn)
        self.ads = pd.read_sql("SELECT * FROM ads", conn)
        conn.close()

        # Pre-compute feature columns for model input
        self._prepare_feature_templates()

    def _prepare_feature_templates(self):
        """Pre-compute lookup tables and establish the canonical feature column order.

        Uses the same pd.get_dummies call as the student notebook to guarantee
        identical column names and ordering.
        """
        # Raw user attributes indexed by user_id
        self.user_raw = self.users_full.set_index("user_id")[[
            "age_group", "gender", "device_type", "region",
            "account_age_days", "past_purchases", "avg_order_value",
            "sessions_per_week", "loyalty_tier",
        ]]

        # Raw ad attributes indexed by ad_id
        self.ad_raw = self.ads.set_index("ad_id")[[
            "category", "ad_format", "product_price", "discount_pct",
            "creative_quality_score", "headline_clickbait_score",
            "brand_familiarity",
        ]]

        # User hidden attributes for the DGP
        self.user_hidden = self.users_full.set_index("user_id")[
            ["_base_click", "_base_convert", "_price_sensitivity"]
        ]

        # Build a diverse reference sample to establish canonical column order.
        # Must include ALL category values so pd.get_dummies creates all columns.
        ref_rows = []
        base_user = self.user_raw.iloc[0].to_dict()
        base_ad = self.ad_raw.iloc[0].to_dict()
        base_ctx = {"page_type": "home", "position": 1, "hour_of_day": 12,
                     "day_of_week": 3, "session_depth": 1}

        # One row per unique value of each categorical column
        for col, source in [
            ("age_group", self.user_raw), ("gender", self.user_raw),
            ("device_type", self.user_raw), ("region", self.user_raw),
            ("loyalty_tier", self.user_raw),
            ("category", self.ad_raw), ("ad_format", self.ad_raw),
        ]:
            for val in source[col].unique():
                row = {**base_user, **base_ad, **base_ctx, col: val}
                ref_rows.append(row)
        for pt in ["home", "search", "product", "category", "checkout"]:
            row = {**base_user, **base_ad, **base_ctx, "page_type": pt}
            ref_rows.append(row)

        ref_encoded = self._encode_raw(pd.DataFrame(ref_rows))
        self.feature_columns = ref_encoded.columns.tolist()

    @staticmethod
    def _encode_raw(df: pd.DataFrame) -> pd.DataFrame:
        """Encode raw features exactly the same way the student notebook does."""
        return pd.get_dummies(
            df[[
                'age_group', 'gender', 'device_type', 'region',
                'account_age_days', 'past_purchases', 'avg_order_value',
                'sessions_per_week', 'loyalty_tier',
                'category', 'ad_format', 'product_price', 'discount_pct',
                'creative_quality_score', 'headline_clickbait_score',
                'brand_familiarity',
                'page_type', 'position', 'hour_of_day', 'day_of_week',
                'session_depth',
            ]],
            columns=['age_group', 'gender', 'device_type', 'region',
                     'loyalty_tier', 'category', 'ad_format', 'page_type'],
            drop_first=True,
            dtype=int,
        )

    def build_validation_dummy(self) -> pd.DataFrame:
        """Build a single-row DataFrame with correct feature columns for model validation."""
        user_id = self.users_full["user_id"].iloc[0]
        ad_id = self.ads["ad_id"].iloc[0]
        return self._build_features(
            user_id, np.array([ad_id]), "search", 2, 14, 3, 3
        )

    def _build_features(self, user_id: int, ad_ids: np.ndarray,
                        page_type: str, position: int,
                        hour: int, day: int, session_depth: int) -> pd.DataFrame:
        """Build the feature DataFrame a model receives, matching student column order."""
        n = len(ad_ids)

        # Build raw (un-encoded) rows
        user = self.user_raw.loc[user_id]
        rows = []
        for ad_id in ad_ids:
            ad = self.ad_raw.loc[ad_id]
            row = {**user.to_dict(), **ad.to_dict(),
                   "page_type": page_type, "position": position,
                   "hour_of_day": hour, "day_of_week": day,
                   "session_depth": session_depth}
            rows.append(row)

        raw_df = pd.DataFrame(rows)
        encoded = self._encode_raw(raw_df)

        # Ensure all canonical columns are present (fill missing with 0)
        # and column order matches training
        return encoded.reindex(columns=self.feature_columns, fill_value=0)

    def _simulate_outcome(self, user_id: int, ad_id: int,
                          page_type: str, position: int,
                          hour: int, session_depth: int) -> tuple[bool, bool, float]:
        """Use the ground-truth DGP to determine click/convert/revenue."""
        u = self.user_hidden.loc[user_id]
        a = self.ads.set_index("ad_id").loc[ad_id]

        effective_price = a["product_price"] * (1 - a["discount_pct"] / 100)

        click_logit = (
            u["_base_click"]
            + 0.5 * a["headline_clickbait_score"] / 10
            + 0.2 * a["creative_quality_score"] / 10
            - 0.6 * (position - 1) / 4
            + 0.3 * (page_type == "search")
            + 0.15 * (page_type == "product")
            - 0.1 * min(session_depth, 10) / 10
            + 0.25 * a["discount_pct"] / 50
            + 0.1 * a["brand_familiarity"] / 10
            - 0.15 * (hour < 6 or hour > 22)
            + self.rng.normal(0, 0.3)
        )
        clicked = self.rng.random() < sigmoid(click_logit)

        if not clicked:
            return False, False, 0.0

        convert_logit = (
            u["_base_convert"]
            + 0.5 * a["creative_quality_score"] / 10
            - 0.4 * a["headline_clickbait_score"] / 10
            - u["_price_sensitivity"] * effective_price / 150
            + 0.25 * a["brand_familiarity"] / 10
            + 0.15 * a["discount_pct"] / 50
            + 0.2 * (page_type == "product")
            + 0.15 * (page_type == "search")
            - 0.1 * (hour < 8 or hour > 21)
            + self.rng.normal(0, 0.25)
        )
        converted = self.rng.random() < sigmoid(convert_logit)

        revenue = 0.0
        if converted:
            revenue = effective_price * self.rng.uniform(0.8, 1.5)

        return clicked, converted, round(revenue, 2)

    def _score_team(self, team_name: str, features: pd.DataFrame) -> tuple[np.ndarray | None, float, bool]:
        """Score all candidate ads using a team's model. Returns (scores, latency_ms, had_error)."""
        model = self.teams[team_name].get("model")
        if model is None:
            return None, 0.0, True

        start = time.perf_counter()
        try:
            scores = model.predict(features)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return np.array(scores, dtype=float), elapsed_ms, False
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return None, elapsed_ms, True

    async def run(self):
        """Main simulation loop."""
        if self.running:
            return
        self.running = True
        self.total_requests = 0

        # Reset all team metrics
        for team in self.teams.values():
            team["metrics"] = {
                "impressions": 0, "clicks": 0, "conversions": 0,
                "revenue": 0.0, "total_latency_ms": 0.0,
                "latency_violations": 0, "errors": 0,
            }
            team["history"] = []

        batch_interval = 1.0  # seconds between batches
        snapshot_interval = 3.0  # seconds between history snapshots
        last_snapshot = time.time()

        print(f"Simulation started: {self.requests_per_second} req/s, "
              f"latency budget {self.latency_budget_ms}ms")

        while self.running:
            batch_size = self.requests_per_second
            batch_start = time.time()

            for _ in range(batch_size):
                if not self.running:
                    break
                await self._process_one_request()
                self.total_requests += 1

            # Save history snapshots
            now = time.time()
            if now - last_snapshot >= snapshot_interval:
                self._save_snapshot()
                last_snapshot = now

            # Sleep to maintain the target rate
            elapsed = time.time() - batch_start
            sleep_time = max(0, batch_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _process_one_request(self):
        """Simulate one incoming user visit and score all teams."""
        # Random user visit
        user_id = int(self.rng.choice(self.users_full["user_id"].values))
        page_type = self.rng.choice(
            ["home", "search", "product", "category", "checkout"],
            p=[0.20, 0.25, 0.20, 0.25, 0.10]
        )
        position = int(self.rng.choice([1, 2, 3, 4, 5], p=[0.20, 0.25, 0.25, 0.20, 0.10]))
        hour = int(self.rng.choice(24))
        day = int(self.rng.choice(7))
        session_depth = int(self.rng.geometric(0.3))

        # Sample a set of candidate ads (simulate an ad auction)
        n_candidates = 10
        candidate_ad_ids = self.rng.choice(
            self.ads["ad_id"].values, size=n_candidates, replace=False
        )

        # Build features once (all teams see the same candidates)
        features = self._build_features(
            user_id, candidate_ad_ids, page_type, position, hour, day, session_depth
        )

        # Score for each team
        for team_name, team in self.teams.items():
            if team["model"] is None:
                continue

            scores, latency_ms, had_error = self._score_team(team_name, features)
            m = team["metrics"]
            m["impressions"] += 1
            m["total_latency_ms"] += latency_ms

            if had_error:
                m["errors"] += 1
                # Fallback: random ad
                chosen_ad_id = int(self.rng.choice(candidate_ad_ids))
            elif latency_ms > self.latency_budget_ms:
                m["latency_violations"] += 1
                # Latency penalty: with probability proportional to overshoot, use random ad
                penalty_prob = 1.0 - (self.latency_budget_ms / latency_ms)
                if self.rng.random() < penalty_prob:
                    chosen_ad_id = int(self.rng.choice(candidate_ad_ids))
                else:
                    chosen_ad_id = int(candidate_ad_ids[np.argmax(scores)])
            else:
                chosen_ad_id = int(candidate_ad_ids[np.argmax(scores)])

            # Simulate the outcome using the ground-truth DGP
            clicked, converted, revenue = self._simulate_outcome(
                user_id, chosen_ad_id, page_type, position, hour, session_depth
            )

            if clicked:
                m["clicks"] += 1
            if converted:
                m["conversions"] += 1
                m["revenue"] += revenue

    def _save_snapshot(self):
        """Save a snapshot of current metrics for time-series plotting."""
        ts = time.time()
        for team in self.teams.values():
            if team["model"] is None:
                continue
            m = team["metrics"]
            team["history"].append({
                "t": ts,
                "impressions": m["impressions"],
                "clicks": m["clicks"],
                "conversions": m["conversions"],
                "revenue": round(m["revenue"], 2),
                "rpi": round(m["revenue"] / max(m["impressions"], 1), 6),
            })

    def stop(self):
        self.running = False
        print("Simulation stopped.")
