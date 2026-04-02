"""
Data generator for the Predict-Ship-Compete workshop.

Creates a SQLite database with realistic e-commerce ad data.
The data has intentional structure that rewards modeling the full funnel
(CTR * CVR * revenue) over naive CTR optimization.

Key hidden segments:
- Window Shoppers (30%): click everything, rarely buy
- Bargain Hunters (25%): respond to discounts, price-sensitive
- Loyal Customers (20%): moderate clicks, high conversion, high AOV
- Impulse Buyers (15%): moderate everything, variable
- Researchers (10%): rarely click, but when they do → big purchases
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_USERS = 10_000
N_ADS = 200
N_IMPRESSIONS = 500_000

DB_PATH = Path(__file__).parent / "data" / "workshop.db"

# --- Segment definitions (the hidden truth students must discover) ---

SEGMENT_CONFIG = {
    "window_shopper": {
        "proportion": 0.30,
        "base_click": -0.8,       # high CTR ~30%
        "base_convert": -3.5,     # very low CVR ~3%
        "price_sensitivity": 0.3,
        "age_dist": [0.35, 0.30, 0.15, 0.10, 0.10],   # skews young
        "device_dist": [0.65, 0.20, 0.15],              # mostly mobile
        "avg_past_purchases": 2,
        "avg_order_value_range": (15, 40),
        "sessions_per_week_range": (8, 20),
    },
    "bargain_hunter": {
        "proportion": 0.25,
        "base_click": -1.6,       # moderate CTR ~17%
        "base_convert": -1.5,     # moderate CVR ~18%
        "price_sensitivity": 2.5,  # VERY price sensitive
        "age_dist": [0.15, 0.25, 0.25, 0.20, 0.15],
        "device_dist": [0.40, 0.40, 0.20],
        "avg_past_purchases": 8,
        "avg_order_value_range": (20, 50),
        "sessions_per_week_range": (4, 10),
    },
    "loyal_customer": {
        "proportion": 0.20,
        "base_click": -2.0,       # lower CTR ~12%
        "base_convert": 0.2,      # high CVR ~55%
        "price_sensitivity": 0.2,  # not price sensitive
        "age_dist": [0.05, 0.15, 0.25, 0.30, 0.25],   # skews older
        "device_dist": [0.25, 0.55, 0.20],              # mostly desktop
        "avg_past_purchases": 30,
        "avg_order_value_range": (60, 150),
        "sessions_per_week_range": (3, 7),
    },
    "impulse_buyer": {
        "proportion": 0.15,
        "base_click": -1.8,
        "base_convert": -0.5,     # decent CVR ~38%
        "price_sensitivity": 1.0,
        "age_dist": [0.20, 0.30, 0.25, 0.15, 0.10],
        "device_dist": [0.50, 0.30, 0.20],
        "avg_past_purchases": 12,
        "avg_order_value_range": (25, 80),
        "sessions_per_week_range": (3, 8),
    },
    "researcher": {
        "proportion": 0.10,
        "base_click": -3.0,       # low CTR ~5%
        "base_convert": 1.2,      # very high CVR ~77%
        "price_sensitivity": 0.4,
        "age_dist": [0.05, 0.10, 0.25, 0.30, 0.30],
        "device_dist": [0.15, 0.65, 0.20],
        "avg_past_purchases": 15,
        "avg_order_value_range": (80, 250),
        "sessions_per_week_range": (1, 4),
    },
}

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS = ["M", "F", "Other"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
REGIONS = ["US-East", "US-West", "US-Central", "EU", "APAC"]
LOYALTY_TIERS = ["bronze", "silver", "gold", "platinum"]

AD_CATEGORIES = ["electronics", "fashion", "home", "sports", "beauty", "food"]
AD_FORMATS = ["banner", "native", "video", "carousel"]
PAGE_TYPES = ["home", "search", "product", "category", "checkout"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def generate_users(rng):
    """Generate user table with hidden segment assignments."""
    users = []
    segment_names = list(SEGMENT_CONFIG.keys())
    segment_probs = [SEGMENT_CONFIG[s]["proportion"] for s in segment_names]

    segments = rng.choice(segment_names, size=N_USERS, p=segment_probs)

    for i, seg_name in enumerate(segments):
        seg = SEGMENT_CONFIG[seg_name]
        user_id = i + 1

        age_group = rng.choice(AGE_GROUPS, p=seg["age_dist"])
        gender = rng.choice(GENDERS, p=[0.48, 0.48, 0.04])
        device_type = rng.choice(DEVICE_TYPES, p=seg["device_dist"])
        region = rng.choice(REGIONS, p=[0.25, 0.25, 0.15, 0.20, 0.15])

        account_age_days = int(rng.exponential(400) + 30)
        past_purchases = max(0, int(rng.poisson(seg["avg_past_purchases"])))
        aov_lo, aov_hi = seg["avg_order_value_range"]
        avg_order_value = round(rng.uniform(aov_lo, aov_hi), 2) if past_purchases > 0 else 0.0
        spw_lo, spw_hi = seg["sessions_per_week_range"]
        sessions_per_week = round(rng.uniform(spw_lo, spw_hi), 1)

        # Loyalty tier based on past purchases
        if past_purchases >= 25:
            loyalty_tier = "platinum"
        elif past_purchases >= 15:
            loyalty_tier = "gold"
        elif past_purchases >= 5:
            loyalty_tier = "silver"
        else:
            loyalty_tier = "bronze"

        users.append({
            "user_id": user_id,
            "age_group": age_group,
            "gender": gender,
            "device_type": device_type,
            "region": region,
            "account_age_days": account_age_days,
            "past_purchases": past_purchases,
            "avg_order_value": avg_order_value,
            "loyalty_tier": loyalty_tier,
            "sessions_per_week": sessions_per_week,
            # Hidden fields for the DGP (not stored in the student-facing DB)
            "_segment": seg_name,
            "_base_click": seg["base_click"] + rng.normal(0, 0.2),
            "_base_convert": seg["base_convert"] + rng.normal(0, 0.2),
            "_price_sensitivity": seg["price_sensitivity"] + rng.normal(0, 0.1),
        })

    return pd.DataFrame(users)


def generate_ads(rng):
    """Generate ad catalog."""
    ads = []
    for i in range(N_ADS):
        ad_id = i + 1
        category = rng.choice(AD_CATEGORIES)
        ad_format = rng.choice(AD_FORMATS)

        # Price depends on category
        price_ranges = {
            "electronics": (30, 500),
            "fashion": (15, 200),
            "home": (20, 300),
            "sports": (10, 150),
            "beauty": (5, 80),
            "food": (5, 50),
        }
        lo, hi = price_ranges[category]
        product_price = round(rng.uniform(lo, hi), 2)
        discount_pct = round(rng.choice([0, 0, 0, 5, 10, 15, 20, 25, 30, 40, 50],
                                         p=[0.30, 0.10, 0.05, 0.10, 0.10, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]), 1)

        # Creative quality: better ads are rarer
        creative_quality_score = round(np.clip(rng.beta(2, 3) * 10, 1, 10), 1)

        # Clickbait score: inversely correlated with creative quality (with noise)
        clickbait_base = 10 - creative_quality_score + rng.normal(0, 1.5)
        headline_clickbait_score = round(np.clip(clickbait_base, 1, 10), 1)

        brand_familiarity = round(np.clip(rng.beta(2, 2) * 10, 1, 10), 1)

        ads.append({
            "ad_id": ad_id,
            "category": category,
            "ad_format": ad_format,
            "product_price": product_price,
            "discount_pct": discount_pct,
            "creative_quality_score": creative_quality_score,
            "headline_clickbait_score": headline_clickbait_score,
            "brand_familiarity": brand_familiarity,
        })

    return pd.DataFrame(ads)


def generate_impressions(users_df, ads_df, rng):
    """Generate impression data with realistic click/convert DGP."""
    # Pre-compute hidden user attributes as arrays for vectorized ops
    user_ids = users_df["user_id"].values
    user_base_click = users_df["_base_click"].values
    user_base_convert = users_df["_base_convert"].values
    user_price_sens = users_df["_price_sensitivity"].values

    ad_ids = ads_df["ad_id"].values
    ad_clickbait = ads_df["headline_clickbait_score"].values
    ad_quality = ads_df["creative_quality_score"].values
    ad_price = ads_df["product_price"].values
    ad_discount = ads_df["discount_pct"].values
    ad_brand = ads_df["brand_familiarity"].values

    # Sample random user-ad pairs
    user_indices = rng.choice(len(user_ids), size=N_IMPRESSIONS,
                              p=users_df["sessions_per_week"].values / users_df["sessions_per_week"].sum())
    ad_indices = rng.choice(len(ad_ids), size=N_IMPRESSIONS)

    # Context features
    positions = rng.choice([1, 2, 3, 4, 5], size=N_IMPRESSIONS, p=[0.20, 0.25, 0.25, 0.20, 0.10])
    page_types = rng.choice(PAGE_TYPES, size=N_IMPRESSIONS, p=[0.20, 0.25, 0.20, 0.25, 0.10])
    hours = rng.choice(24, size=N_IMPRESSIONS,
                       p=np.array([1,1,1,1,1,1, 2,3,4,5,5,5, 5,5,5,4,4,4, 4,5,5,4,3,2],
                                  dtype=float) / 80)
    days = rng.choice(7, size=N_IMPRESSIONS)
    session_depths = rng.geometric(0.3, size=N_IMPRESSIONS)

    # Timestamps spanning 90 days
    base_ts = pd.Timestamp("2025-10-01")
    random_seconds = rng.integers(0, 90 * 86400, size=N_IMPRESSIONS)
    random_seconds.sort()  # roughly chronological
    timestamps = pd.to_datetime(base_ts) + pd.to_timedelta(random_seconds, unit="s")

    # --- Click DGP ---
    click_logit = (
        user_base_click[user_indices]
        + 0.5 * ad_clickbait[ad_indices] / 10        # clickbait boosts CTR
        + 0.2 * ad_quality[ad_indices] / 10           # quality helps a bit
        - 0.6 * (positions - 1) / 4                    # position 1 is best
        + 0.3 * (page_types == "search").astype(float) # search = higher intent
        + 0.15 * (page_types == "product").astype(float)
        - 0.1 * np.minimum(session_depths, 10) / 10   # fatigue
        + 0.25 * ad_discount[ad_indices] / 50          # discounts attract clicks
        + 0.1 * ad_brand[ad_indices] / 10              # familiar brands get clicks
        - 0.15 * ((hours < 6) | (hours > 22)).astype(float)  # late night penalty
        + 0.1 * (ad_clickbait[ad_indices] / 10) * (users_df["_segment"].values[user_indices] == "window_shopper").astype(float)  # window shoppers love clickbait
        + rng.normal(0, 0.3, size=N_IMPRESSIONS)       # noise
    )
    click_prob = sigmoid(click_logit)
    clicked = rng.binomial(1, click_prob)

    # --- Conversion DGP (only matters for clicked impressions) ---
    effective_price = ad_price[ad_indices] * (1 - ad_discount[ad_indices] / 100)

    convert_logit = (
        user_base_convert[user_indices]
        + 0.5 * ad_quality[ad_indices] / 10            # quality drives conversion
        - 0.4 * ad_clickbait[ad_indices] / 10          # clickbait HURTS conversion
        - user_price_sens[user_indices] * effective_price / 150  # price sensitivity
        + 0.25 * ad_brand[ad_indices] / 10             # brand trust helps
        + 0.15 * ad_discount[ad_indices] / 50          # discounts help convert
        + 0.2 * (page_types == "product").astype(float) # product page = more intent
        + 0.15 * (page_types == "search").astype(float)
        - 0.1 * ((hours < 8) | (hours > 21)).astype(float)  # off-hours less likely to buy
        + rng.normal(0, 0.25, size=N_IMPRESSIONS)
    )
    convert_prob = sigmoid(convert_logit) * clicked  # only clicked items can convert
    converted = rng.binomial(1, convert_prob)

    # Revenue for conversions
    revenue = np.where(converted == 1, effective_price, 0.0)
    # Add some variance to revenue (upsells, quantity variation)
    revenue = np.where(revenue > 0, revenue * rng.uniform(0.8, 1.5, size=N_IMPRESSIONS), 0.0)
    revenue = np.round(revenue, 2)

    # Build impressions DataFrame
    impressions = pd.DataFrame({
        "impression_id": np.arange(1, N_IMPRESSIONS + 1),
        "user_id": user_ids[user_indices],
        "ad_id": ad_ids[ad_indices],
        "timestamp": timestamps,
        "page_type": page_types,
        "position": positions,
        "day_of_week": days,
        "hour_of_day": hours,
        "session_depth": session_depths,
        "clicked": clicked,
    })

    # Build conversions DataFrame (only converted impressions)
    conv_mask = converted == 1
    conversions = pd.DataFrame({
        "conversion_id": np.arange(1, conv_mask.sum() + 1),
        "impression_id": impressions.loc[conv_mask, "impression_id"].values,
        "user_id": user_ids[user_indices[conv_mask]],
        "ad_id": ad_ids[ad_indices[conv_mask]],
        "revenue": revenue[conv_mask],
        "time_to_convert_minutes": np.round(rng.exponential(30, size=conv_mask.sum()), 1),
    })

    return impressions, conversions, click_prob, convert_prob


def print_summary(users_df, ads_df, impressions_df, conversions_df):
    """Print data summary for verification."""
    n_clicks = impressions_df["clicked"].sum()
    n_convs = len(conversions_df)
    total_rev = conversions_df["revenue"].sum()

    print(f"\n{'='*60}")
    print(f"  Workshop Database Generated Successfully")
    print(f"{'='*60}")
    print(f"  Users:        {len(users_df):>10,}")
    print(f"  Ads:          {len(ads_df):>10,}")
    print(f"  Impressions:  {len(impressions_df):>10,}")
    print(f"  Clicks:       {n_clicks:>10,}  (CTR: {n_clicks/len(impressions_df):.1%})")
    print(f"  Conversions:  {n_convs:>10,}  (CVR|click: {n_convs/max(n_clicks,1):.1%})")
    print(f"  Total Revenue: ${total_rev:>12,.2f}")
    print(f"{'='*60}")

    # Per-segment breakdown
    merged = impressions_df.merge(
        users_df[["user_id", "_segment"]], on="user_id"
    )
    conv_user = conversions_df.merge(
        users_df[["user_id", "_segment"]], on="user_id"
    )

    print(f"\n  Per-Segment Breakdown (hidden from students):")
    print(f"  {'Segment':<20} {'Users':>6} {'Impressions':>12} {'CTR':>8} {'CVR|click':>10} {'Revenue':>12} {'Rev/Impr':>10}")
    print(f"  {'-'*80}")

    for seg in SEGMENT_CONFIG:
        seg_imp = merged[merged["_segment"] == seg]
        seg_conv = conv_user[conv_user["_segment"] == seg]
        n_imp = len(seg_imp)
        n_clk = seg_imp["clicked"].sum()
        n_cv = len(seg_conv)
        rev = seg_conv["revenue"].sum()
        n_users = (users_df["_segment"] == seg).sum()

        print(f"  {seg:<20} {n_users:>6,} {n_imp:>12,} {n_clk/max(n_imp,1):>8.1%} "
              f"{n_cv/max(n_clk,1):>10.1%} ${rev:>11,.2f} ${rev/max(n_imp,1):>9.4f}")

    print()


def save_to_sqlite(users_df, ads_df, impressions_df, conversions_df):
    """Save data to SQLite, excluding hidden columns."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    # Users: drop hidden columns
    users_public = users_df.drop(columns=[c for c in users_df.columns if c.startswith("_")])
    users_public.to_sql("users", conn, index=False)

    ads_df.to_sql("ads", conn, index=False)
    impressions_df.to_sql("impressions", conn, index=False)
    conversions_df.to_sql("conversions", conn, index=False)

    # Create indexes for performance
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX idx_impressions_user ON impressions(user_id)")
    cursor.execute("CREATE INDEX idx_impressions_ad ON impressions(ad_id)")
    cursor.execute("CREATE INDEX idx_impressions_clicked ON impressions(clicked)")
    cursor.execute("CREATE INDEX idx_conversions_impression ON conversions(impression_id)")
    cursor.execute("CREATE INDEX idx_conversions_user ON conversions(user_id)")
    cursor.execute("CREATE INDEX idx_conversions_ad ON conversions(ad_id)")
    conn.commit()

    # Also save the full users table (with hidden cols) for the simulator
    users_df.to_sql("_users_full", conn, index=False)
    cursor.execute("CREATE INDEX idx_users_full_id ON _users_full(user_id)")
    conn.commit()

    conn.close()
    print(f"  Database saved to: {DB_PATH}")
    print(f"  Size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    rng = np.random.default_rng(SEED)

    print("Generating users...")
    users_df = generate_users(rng)

    print("Generating ads...")
    ads_df = generate_ads(rng)

    print("Generating impressions & conversions...")
    impressions_df, conversions_df, _, _ = generate_impressions(users_df, ads_df, rng)

    print_summary(users_df, ads_df, impressions_df, conversions_df)
    save_to_sqlite(users_df, ads_df, impressions_df, conversions_df)


if __name__ == "__main__":
    main()
