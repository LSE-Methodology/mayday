"""Concurrent stress test for the Predict-Ship-Compete server.

Simulates N students hitting the server simultaneously across the workshop's
main load patterns:

  1. health check
  2. concurrent registration burst
  3. concurrent SQL data pulls (~500k rows each, in 50k batches)
  4. train one shared CTR model
  5. concurrent model uploads (validation runs on the server's event loop)
  6. live simulation + concurrent dashboard polling

In default mode every team uploads the same small model — that exercises
upload throughput but not real-world variety. Pass `--chaos` to upload a
diverse set of intentionally-pathological models (huge RF, sleepy wrappers,
NaN returners, oversize pickles, predict-raises) and watch how the simulator
copes.

Run from a separate terminal while `python app.py` is running. Defaults to 20
fake teams against localhost — both overridable via CLI.

Usage:
  python stress_test.py
  python stress_test.py --teams 30 --sim-duration 120
  python stress_test.py --server http://10.0.0.5:8000
  python stress_test.py --stages register,upload,sim --rows-per-student 50000
  python stress_test.py --chaos --sim-duration 120

Notes:
  - Stress runs use timestamp-prefixed team names so they do not collide with
    real workshop teams. The server's 20-team cap is per-process: restart the
    server (`python app.py`) before the real workshop so the cap isn't hit.
  - For a true reachability/firewall test, run this script from a peer machine
    against the instructor laptop's IP, not localhost.
"""

import argparse
import asyncio
import io
import time

import cloudpickle
import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd


HTTP_TIMEOUT = 120.0


class CTRScoringModel:
    """Wraps a probabilistic classifier so .predict() returns P(class=1)."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class SleepyModel:
    """A model that artificially pads inference time. Tests the sim's
    behaviour under a slow-by-mistake student model."""

    def __init__(self, inner, sleep_s: float):
        self.inner = inner
        self.sleep_s = sleep_s

    def predict(self, X):
        time.sleep(self.sleep_s)
        return self.inner.predict(X)


class NaNModel:
    """Returns NaN for every score. np.argmax silently picks index 0."""

    def predict(self, X):
        return np.full(len(X), np.nan)


class RaisingModel:
    """Always raises in predict(). Should be rejected by upload validation."""

    def predict(self, X):
        raise RuntimeError("intentional failure for stress test")


class WrongShapeModel:
    """Returns a scalar instead of a per-row array. Should be rejected."""

    def predict(self, X):
        return 0.5


class OversizedModel:
    """A valid model carrying a large random byte payload — used to confirm
    that the server enforces its upload size limit (default 50MB)."""

    def __init__(self, padding_mb: int):
        import os
        self.pad = os.urandom(padding_mb * 1024 * 1024)

    def predict(self, X):
        return np.zeros(len(X))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        df[['age_group', 'gender', 'device_type', 'region',
            'account_age_days', 'past_purchases', 'avg_order_value',
            'sessions_per_week', 'loyalty_tier',
            'category', 'ad_format', 'product_price', 'discount_pct',
            'creative_quality_score', 'headline_clickbait_score',
            'brand_familiarity',
            'page_type', 'position', 'hour_of_day', 'day_of_week',
            'session_depth']],
        columns=['age_group', 'gender', 'device_type', 'region',
                 'loyalty_tier', 'category', 'ad_format', 'page_type'],
        drop_first=True, dtype=int,
    )


def report(label: str, durations_ms: list[float], errors: int) -> None:
    n = len(durations_ms)
    if n == 0:
        print(f"  {label}: 0 ok, {errors} errors")
        return
    arr = np.asarray(durations_ms)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"  {label}: n={n} p50={p50:.0f}ms p95={p95:.0f}ms "
          f"p99={p99:.0f}ms max={arr.max():.0f}ms errors={errors}")


async def health_check(client: httpx.AsyncClient, server: str) -> None:
    print("\n=== Stage 1: Health check ===")
    r = await client.get(f"{server}/api/schema")
    r.raise_for_status()
    tables = r.json()["tables"]
    for name, info in tables.items():
        print(f"  {name}: {info['row_count']:,} rows")


async def register_burst(client: httpx.AsyncClient, server: str,
                          team_names: list[str]) -> None:
    print(f"\n=== Stage 2: Concurrent registration ({len(team_names)} teams) ===")

    async def reg(name: str):
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{server}/api/teams/{name}/register",
                json={"members": ["stress"]},
            )
            r.raise_for_status()
            return (time.perf_counter() - t0) * 1000, None
        except Exception as e:
            return (time.perf_counter() - t0) * 1000, e

    results = await asyncio.gather(*[reg(n) for n in team_names])
    durations = [dt for dt, err in results if err is None]
    errors = [err for _, err in results if err is not None]
    report("register", durations, len(errors))
    if errors:
        print(f"  Sample errors: {errors[:3]}")


_TRAIN_SQL = """
SELECT u.age_group, u.gender, u.device_type, u.region,
       u.account_age_days, u.past_purchases, u.avg_order_value,
       u.sessions_per_week, u.loyalty_tier,
       a.category, a.ad_format, a.product_price, a.discount_pct,
       a.creative_quality_score, a.headline_clickbait_score, a.brand_familiarity,
       i.page_type, i.position, i.hour_of_day, i.day_of_week, i.session_depth,
       i.clicked,
       CASE WHEN c.conversion_id IS NOT NULL THEN 1 ELSE 0 END as converted,
       COALESCE(c.revenue, 0) as revenue
FROM impressions i
JOIN users u ON i.user_id = u.user_id
JOIN ads a ON i.ad_id = a.ad_id
LEFT JOIN conversions c ON i.impression_id = c.impression_id
ORDER BY i.impression_id
LIMIT {batch} OFFSET {offset}
"""


async def sql_flood(client: httpx.AsyncClient, server: str,
                     n_students: int, rows_per_student: int) -> None:
    print(f"\n=== Stage 3: Concurrent SQL pulls "
          f"({n_students} students × {rows_per_student:,} rows) ===")
    BATCH = 50_000

    async def student_pull(sid: int) -> tuple[list[float], int, int]:
        durations = []
        errors = 0
        rows = 0
        for offset in range(0, rows_per_student, BATCH):
            batch = min(BATCH, rows_per_student - offset)
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{server}/api/sql", json={
                    "query": _TRAIN_SQL.format(batch=batch, offset=offset),
                    "limit": batch,
                })
                r.raise_for_status()
                rows += len(r.json()["rows"])
                durations.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1
        return durations, errors, rows

    t0 = time.perf_counter()
    results = await asyncio.gather(*[student_pull(i) for i in range(n_students)])
    wall = time.perf_counter() - t0
    durations = [d for r in results for d in r[0]]
    errors = sum(r[1] for r in results)
    total_rows = sum(r[2] for r in results)
    report("sql request", durations, errors)
    print(f"  Wall: {wall:.1f}s  total rows: {total_rows:,}  "
          f"~{total_rows / max(wall, 0.01):,.0f} rows/s aggregate")


async def train_shared_model(client: httpx.AsyncClient, server: str,
                               train_rows: int = 100_000) -> bytes:
    print(f"\n=== Stage 4: Train shared CTR model ({train_rows:,} rows) ===")
    BATCH = 50_000
    frames = []
    t0 = time.perf_counter()
    for offset in range(0, train_rows, BATCH):
        batch = min(BATCH, train_rows - offset)
        r = await client.post(f"{server}/api/sql", json={
            "query": _TRAIN_SQL.format(batch=batch, offset=offset),
            "limit": batch,
        })
        r.raise_for_status()
        d = r.json()
        frames.append(pd.DataFrame(d["rows"], columns=d["columns"]))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Pulled {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

    X = prepare_features(df)
    y = df["clicked"].values

    t0 = time.perf_counter()
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        num_leaves=31, verbose=-1,
    )
    model.fit(X, y)
    print(f"  Trained LightGBM in {time.perf_counter() - t0:.1f}s")

    buf = io.BytesIO()
    cloudpickle.dump(CTRScoringModel(model), buf)
    print(f"  Pickled model size: {buf.tell() / 1024:.1f} KB")
    return buf.getvalue()


async def _pull_train_df(client: httpx.AsyncClient, server: str,
                          train_rows: int) -> pd.DataFrame:
    BATCH = 50_000
    frames = []
    for offset in range(0, train_rows, BATCH):
        batch = min(BATCH, train_rows - offset)
        r = await client.post(f"{server}/api/sql", json={
            "query": _TRAIN_SQL.format(batch=batch, offset=offset),
            "limit": batch,
        })
        r.raise_for_status()
        d = r.json()
        frames.append(pd.DataFrame(d["rows"], columns=d["columns"]))
    return pd.concat(frames, ignore_index=True)


def _pickle(obj) -> bytes:
    buf = io.BytesIO()
    cloudpickle.dump(obj, buf)
    return buf.getvalue()


async def build_chaos_variants(client: httpx.AsyncClient, server: str,
                                 train_rows: int = 50_000) -> list[tuple[str, bytes]]:
    """Train a small data set, then build a diverse set of pickled models.

    Mix includes: cheap LR, small/medium/big LightGBM, RF up to ~10MB, an
    oversized payload for the upload limit, two artificially-slow wrappers,
    and three pathological models to confirm the validation step rejects
    them.
    """
    print(f"\n=== Chaos: build diverse model set ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    df = await _pull_train_df(client, server, train_rows)
    X = prepare_features(df)
    y = df["clicked"].values
    print(f"  Trained on {len(X):,} rows × {X.shape[1]} features")

    variants: list[tuple[str, bytes]] = []
    t0 = time.perf_counter()

    def add(name: str, payload: bytes) -> None:
        variants.append((name, payload))
        print(f"    {name:<14} {len(payload) / 1024:>10.1f} KB")

    lr_small = LogisticRegression(max_iter=200, solver="liblinear").fit(X, y)
    add("tiny-lr", _pickle(CTRScoringModel(lr_small)))

    lr_l2 = LogisticRegression(max_iter=500, C=0.1, solver="lbfgs").fit(X, y)
    add("small-lr", _pickle(CTRScoringModel(lr_l2)))

    tiny_lgb = lgb.LGBMClassifier(n_estimators=30, max_depth=4, verbose=-1).fit(X, y)
    add("tiny-lgb", _pickle(CTRScoringModel(tiny_lgb)))

    medium_lgb = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, num_leaves=31, verbose=-1,
    ).fit(X, y)
    add("medium-lgb", _pickle(CTRScoringModel(medium_lgb)))

    big_lgb = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=10, num_leaves=127, verbose=-1,
    ).fit(X, y)
    add("big-lgb", _pickle(CTRScoringModel(big_lgb)))

    small_rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, n_jobs=1, random_state=0,
    ).fit(X, y)
    add("small-rf", _pickle(CTRScoringModel(small_rf)))

    huge_rf = RandomForestClassifier(
        n_estimators=200, max_depth=18, n_jobs=1, random_state=0,
    ).fit(X, y)
    add("huge-rf", _pickle(CTRScoringModel(huge_rf)))

    # Should be rejected by the 50 MB upload limit
    add("oversize", _pickle(OversizedModel(60)))

    # Pass validation but be slow at sim time — the highest-risk class
    add("sleepy-50ms", _pickle(SleepyModel(CTRScoringModel(lr_small), 0.05)))
    add("sleepy-200ms", _pickle(SleepyModel(CTRScoringModel(lr_small), 0.20)))

    # Pass validation, but predict() returns NaN — sim should keep running
    add("returns-nan", _pickle(NaNModel()))

    # Should be rejected at validation
    add("raises", _pickle(RaisingModel()))
    add("wrong-shape", _pickle(WrongShapeModel()))

    print(f"  Built {len(variants)} variants in {time.perf_counter() - t0:.1f}s")
    return variants


async def chaos_upload(client: httpx.AsyncClient, server: str, prefix: str,
                        variants: list[tuple[str, bytes]]) -> list[str]:
    """Register one team per variant, then upload each variant concurrently.

    Reports which uploads were accepted vs rejected and why. Returns the team
    names whose models were accepted (i.e. will participate in the sim).
    """
    print(f"\n=== Chaos: register + upload {len(variants)} variants ===")
    team_names = [f"{prefix}-{name}" for name, _ in variants]

    reg_results = await asyncio.gather(*[
        client.post(f"{server}/api/teams/{n}/register", json={"members": ["chaos"]})
        for n in team_names
    ], return_exceptions=True)
    reg_failed = [r for r in reg_results if isinstance(r, Exception) or r.status_code >= 400]
    if reg_failed:
        print(f"  WARN: {len(reg_failed)} registrations failed (server team cap?). "
              "Restart `python app.py` between runs.")

    async def upload(variant: tuple[str, bytes]):
        name, payload = variant
        team = f"{prefix}-{name}"
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{server}/api/teams/{team}/model",
                files={"model": ("model.pkl", payload, "application/octet-stream")},
            )
            dt = (time.perf_counter() - t0) * 1000
            body = r.json() if "application/json" in r.headers.get("content-type", "") else {}
            return name, team, r.status_code, dt, body
        except Exception as e:
            return name, team, None, (time.perf_counter() - t0) * 1000, {"detail": str(e)}

    results = await asyncio.gather(*[upload(v) for v in variants])
    print(f"  {'variant':<14} {'status':<10} {'rt (ms)':>8}  detail")
    accepted = []
    for name, team, status, dt, body in results:
        if status == 200:
            val = body.get("validation_latency_ms", "?")
            print(f"    {name:<14} {'OK':<10} {dt:>8.0f}  val_lat={val}ms")
            accepted.append(team)
        else:
            detail = body.get("detail", body) if isinstance(body, dict) else body
            label = f"HTTP {status}" if status else "FAILED"
            print(f"    {name:<14} {label:<10} {dt:>8.0f}  {str(detail)[:90]}")
    return accepted


async def chaos_sim(client: httpx.AsyncClient, server: str,
                     accepted_teams: list[str], duration_s: int,
                     requests_per_second: int, latency_budget_ms: int) -> None:
    print(f"\n=== Chaos: live sim with {len(accepted_teams)} models "
          f"({duration_s}s @ {requests_per_second}/s, budget {latency_budget_ms}ms) ===")
    r = await client.post(
        f"{server}/api/simulation/start",
        params={"requests_per_second": requests_per_second,
                "latency_budget_ms": latency_budget_ms},
    )
    r.raise_for_status()

    # Mid-run snapshot so a stuck/slow team is visible before the end
    await asyncio.sleep(min(duration_s / 2, 30))
    mid = (await client.get(f"{server}/api/leaderboard")).json()
    print(f"  Mid-run: {mid['total_requests']:,} requests so far")

    await asyncio.sleep(duration_s - min(duration_s / 2, 30))
    final = (await client.get(f"{server}/api/leaderboard")).json()
    await client.post(f"{server}/api/simulation/stop")

    target = requests_per_second * duration_s
    actual_rps = final["total_requests"] / duration_s
    print(f"  Sim processed {final['total_requests']:,} requests in {duration_s}s "
          f"(target {target:,}, actual {actual_rps:.1f}/s)")
    if actual_rps < requests_per_second * 0.5:
        print(f"  WARN: sim ran at <50% target — slow models drag the whole loop "
              f"because simulator.py:280 scores teams sequentially")

    chaos_set = set(accepted_teams)
    rows = [t for t in final["leaderboard"] if t["team"] in chaos_set]
    rows.sort(key=lambda x: x["avg_latency_ms"], reverse=True)

    print(f"\n  Per-variant breakdown (sorted by avg latency):")
    print(f"    {'team':<32} {'impr':>7} {'avg_lat':>10} "
          f"{'lat_viol':>9} {'errors':>7} {'RPI':>9}")
    for t in rows:
        print(f"    {t['team']:<32} {t['impressions']:>7,} "
              f"{t['avg_latency_ms']:>8.1f}ms {t['latency_violations']:>9} "
              f"{t['errors']:>7} ${t['revenue_per_impression']:>7.4f}")


async def upload_burst(client: httpx.AsyncClient, server: str,
                        team_names: list[str], model_bytes: bytes) -> None:
    print(f"\n=== Stage 5: Concurrent model uploads ({len(team_names)} teams) ===")

    async def upload(name: str):
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{server}/api/teams/{name}/model",
                files={"model": ("model.pkl", model_bytes,
                                  "application/octet-stream")},
            )
            r.raise_for_status()
            return (time.perf_counter() - t0) * 1000, r.json(), None
        except Exception as e:
            return (time.perf_counter() - t0) * 1000, None, e

    results = await asyncio.gather(*[upload(n) for n in team_names])
    durations = [dt for dt, _, err in results if err is None]
    server_validations = [
        body["validation_latency_ms"]
        for _, body, err in results
        if err is None and body and "validation_latency_ms" in body
    ]
    errors = [err for _, _, err in results if err is not None]
    report("upload (round-trip)", durations, len(errors))
    if server_validations:
        report("upload (server validation)", server_validations, 0)
    if errors:
        print(f"  Sample errors: {errors[:3]}")


async def sim_with_pollers(client: httpx.AsyncClient, server: str,
                            n_pollers: int, duration_s: int,
                            requests_per_second: int,
                            latency_budget_ms: int) -> None:
    print(f"\n=== Stage 6: Simulation + dashboard pollers "
          f"({n_pollers} pollers, {duration_s}s, {requests_per_second}/s) ===")
    r = await client.post(
        f"{server}/api/simulation/start",
        params={"requests_per_second": requests_per_second,
                "latency_budget_ms": latency_budget_ms},
    )
    r.raise_for_status()
    start_body = r.json()
    print(f"  Started with {len(start_body.get('teams', []))} teams that have models")

    stop_at = time.perf_counter() + duration_s

    async def poll(pid: int) -> tuple[list[float], int]:
        durations = []
        errors = 0
        while time.perf_counter() < stop_at:
            t0 = time.perf_counter()
            try:
                r = await client.get(f"{server}/api/leaderboard")
                r.raise_for_status()
                durations.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1
            await asyncio.sleep(1.0)
        return durations, errors

    t0 = time.perf_counter()
    results = await asyncio.gather(*[poll(i) for i in range(n_pollers)])
    wall = time.perf_counter() - t0
    durations = [d for r in results for d in r[0]]
    errors = sum(r[1] for r in results)

    final = (await client.get(f"{server}/api/leaderboard")).json()
    await client.post(f"{server}/api/simulation/stop")

    target = requests_per_second * duration_s
    print(f"  Sim processed {final['total_requests']:,} requests in ~{wall:.1f}s "
          f"(target {target:,} = {requests_per_second}/s × {duration_s}s)")
    if final["total_requests"] < target * 0.8:
        print("  WARN: simulation throughput well below target — "
              "per-team model scoring is the likely bottleneck")
    report("dashboard poll", durations, errors)

    print("  Top 3 teams (by RPI):")
    for t in final["leaderboard"][:3]:
        print(f"    {t['team']:<28} RPI=${t['revenue_per_impression']:.4f}  "
              f"avg_lat={t['avg_latency_ms']:.1f}ms  errors={t['errors']}  "
              f"latency_violations={t['latency_violations']}")
    sim_errors = sum(t["errors"] for t in final["leaderboard"])
    sim_violations = sum(t["latency_violations"] for t in final["leaderboard"])
    print(f"  Sim totals: errors={sim_errors}  latency_violations={sim_violations}")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--teams", type=int, default=20)
    p.add_argument("--rows-per-student", type=int, default=500_000,
                   help="Rows each fake student pulls in stage 3")
    p.add_argument("--sim-duration", type=int, default=60,
                   help="Seconds to run the live simulation in stage 6")
    p.add_argument("--rps", type=int, default=20,
                   help="Simulation requests/sec")
    p.add_argument("--latency-budget", type=int, default=50)
    p.add_argument("--prefix", default=None,
                   help="Team-name prefix (default: stress<timestamp>)")
    p.add_argument("--stages", default="all",
                   help="Comma-sep subset of: health,register,sql,train,upload,sim")
    p.add_argument("--chaos", action="store_true",
                   help="Replace stages 4+5 with a diverse-model chaos test "
                        "(13 model variants; sim runs as stage 6).")
    args = p.parse_args()

    prefix = args.prefix or f"stress{int(time.time())}"
    team_names = [f"{prefix}-{i:02d}" for i in range(args.teams)]
    all_stages = ["health", "register", "sql", "train", "upload", "sim"]
    stages = all_stages if args.stages == "all" else args.stages.split(",")
    stages = [s for s in all_stages if s in stages]

    print(f"Server : {args.server}")
    print(f"Teams  : {args.teams}  ({team_names[0]} .. {team_names[-1]})")
    print(f"Stages : {stages}")

    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=limits) as client:
        if "health" in stages:
            await health_check(client, args.server)

        if args.chaos:
            variants = await build_chaos_variants(client, args.server)
            accepted = await chaos_upload(client, args.server, prefix, variants)
            if "sim" in stages and accepted:
                await chaos_sim(
                    client, args.server, accepted,
                    duration_s=args.sim_duration,
                    requests_per_second=args.rps,
                    latency_budget_ms=args.latency_budget,
                )
        else:
            if "register" in stages:
                await register_burst(client, args.server, team_names)
            if "sql" in stages:
                await sql_flood(client, args.server, args.teams, args.rows_per_student)

            model_bytes = None
            if "train" in stages or "upload" in stages:
                model_bytes = await train_shared_model(client, args.server)
            if "upload" in stages and model_bytes is not None:
                await upload_burst(client, args.server, team_names, model_bytes)

            if "sim" in stages:
                await sim_with_pollers(
                    client, args.server,
                    n_pollers=args.teams,
                    duration_s=args.sim_duration,
                    requests_per_second=args.rps,
                    latency_budget_ms=args.latency_budget,
                )

    print("\nDone. Restart the server before the real workshop "
          "so the 20-team cap isn't hit by stress teams.")


if __name__ == "__main__":
    asyncio.run(main())
