# Project MayDay — Instructor Setup Guide

## Live workshop deployment

The server is deployed at **https://lse-mayday.onrender.com** (Render Standard tier). Students point the notebook at this URL; you only need a local install if you want to test or modify the platform.

**Critical: Python version must match across the deployment, your machine, and every student's machine.** The server runs Python 3.12 (pinned in `Dockerfile`). Pickled scikit-learn / LightGBM models cannot be loaded across Python minor versions — students on 3.11, 3.13, etc. will see cryptic "lasti is not an int" errors at upload. Tell them to create a 3.12 environment before the workshop:

```bash
conda create -n mayday python=3.12
conda activate mayday
pip install -r requirements.txt
```

## Quick start (local test run)

```bash
cd mayday

# 1. Create a virtual environment (Python 3.12 to match the deployed server)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the database (takes ~10 seconds)
python generate_data.py

# 4. Start the server
python app.py
```

The server runs at **http://localhost:8000**. Open it in your browser — you should see the landing page.

### Check everything works

| What | URL |
|------|-----|
| Landing page | http://localhost:8000 |
| SQL Explorer | http://localhost:8000/sql |
| Live Dashboard | http://localhost:8000/dashboard |
| API health check | http://localhost:8000/api/schema |

Try running a query in the SQL Explorer:

```sql
SELECT COUNT(*) as impressions, SUM(clicked) as clicks FROM impressions;
```

You should see 500,000 impressions and ~148K clicks.

---

## Simulate the full student experience

With the server running, open a new terminal:

```bash
cd mayday
source .venv/bin/activate
cd notebooks
jupyter notebook student_workbook.ipynb
```

Walk through the notebook as a student would:

1. Set `SERVER = "http://localhost:8000"`
2. Set `TEAM_NAME = "test-team"`
3. Run the cells in order — register, explore, train, wrap, upload
4. Go to http://localhost:8000/dashboard
5. Click **Start Simulation** (or use the API: `curl -X POST http://localhost:8000/api/simulation/start`)
6. Watch the dashboard update in real time
7. Click **Stop** when you've seen enough

You can register a second team and upload a different model to see them compared side by side.

---

## Day-of checklist

### Before students arrive

- [ ] Decide where to run the server (your laptop, a department machine, or cloud — see below)
- [ ] Run `python generate_data.py` to create a fresh database
- [ ] Start the server: `python app.py`
- [ ] Open the dashboard in a browser tab you can project: `http://<host>:8000/dashboard`
- [ ] Test the SQL Explorer and API from a different device on the same network
- [ ] Distribute the student notebook (`notebooks/student_workbook.ipynb`)
- [ ] Share the server URL with students

### During the session

| Time | Phase | What to do |
|------|-------|------------|
| 0:00 | Start | Share server URL. Students open notebooks and register teams. |
| 0:00–0:45 | Explore | Students query the database. Circulate and prompt them to look beyond CTR. |
| 0:45–2:00 | Build | Students pull data and train models. Encourage different target choices across teams. |
| 2:00–2:30 | Optimise | Students benchmark latency. Mention the latency penalty. |
| 2:30–2:45 | Deploy | Students upload models. Check the dashboard — all teams should appear. |
| 2:45 | Go live | Click **Start Simulation** on the dashboard (or adjust rate/latency budget first). |
| 2:45–3:30 | Compete & iterate | Project the dashboard. Students can re-upload improved models. |
| 3:30 | Wrap up | Stop the simulation. Discuss results — why did the winner win? |

### Simulation settings

On the dashboard, before clicking Start:

- **Requests/sec**: 20 is a good default. Increase to 50+ for faster results if you're short on time.
- **Latency budget (ms)**: 50ms is the default. This penalises large ensemble models. Set to 200ms if you want to remove latency as a factor.

---

## Hosting options

### Option A: Your laptop on the local network

Simplest option. Students must be on the same WiFi.

```bash
python app.py  # Listens on 0.0.0.0:8000
```

Find your IP:
```bash
# macOS
ipconfig getifaddr en0

# Linux
hostname -I | awk '{print $1}'
```

Share `http://<your-ip>:8000` with students. Make sure port 8000 isn't blocked by a firewall.

### Option B: Cloud VM (EC2, GCP, etc.)

```bash
# On the VM
pip install -r requirements.txt
python generate_data.py
python app.py
```

Open port 8000 in the security group / firewall rules. Share the public IP with students.

### Option C: Docker

```bash
docker build -t mayday .
docker run -p 8000:8000 mayday
```

(You'll need to add a Dockerfile — a simple one would use `python:3.11-slim`, copy the project, install requirements, run generate_data.py, then CMD python app.py.)

---

## Troubleshooting

**Students can't connect to the server**
- Check they're on the same network (or the port is publicly accessible)
- Check no firewall is blocking port 8000
- Try `curl http://<host>:8000/api/schema` from their machine

**Model upload fails with "feature names" error**
- The model was trained with different column names than the simulator expects
- Students must use the `prepare_features()` function from the notebook, which calls `pd.get_dummies` with the same column spec the simulator uses
- Most common fix: re-run the feature preparation cell and retrain

**Model upload fails with "Failed to unpickle"**
- Students must use `cloudpickle.dump()`, not `pickle.dump()`
- `cloudpickle` captures custom class definitions (like `ScoringModel`) in the pickle file

**Simulation is slow or jerky**
- Reduce requests/sec on the dashboard
- A complex model (large random forest, deep neural net) in one team can slow the whole loop since teams are scored sequentially

**Want to reset everything**
- Stop the server, delete `data/workshop.db`, re-run `python generate_data.py`, restart

---

## What the data looks like (instructor only)

The database has five hidden user segments that students should discover through exploration:

| Segment | Share | CTR | CVR given click | Rev / Impression |
|---------|-------|-----|-----------------|-----------------|
| Window Shoppers | 30% | ~39% | ~3% | ~$1.19 |
| Bargain Hunters | 25% | ~22% | ~9% | ~$1.05 |
| Impulse Buyers | 15% | ~19% | ~30% | ~$4.41 |
| Researchers | 10% | ~6% | ~74% | ~$5.38 |
| Loyal Customers | 20% | ~16% | ~55% | ~$9.37 |

The key insight: **loyal customers and researchers generate 4–8x more revenue per impression than window shoppers**, despite having much lower CTR. A pure CTR model targets window shoppers with clickbait — lots of clicks, almost no revenue.

The data also encodes that high `headline_clickbait_score` boosts P(click) but reduces P(convert|click). So clickbait is a net negative for revenue.

In the end-to-end test, a revenue-predicting model generated **2.7x more revenue** than a CTR model on the same traffic.
