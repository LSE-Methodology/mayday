# Predict, Ship, Compete

A 3-4 hour interactive workshop where students combine SQL, machine learning, and causal inference to build and deploy ad-targeting models in a live A/B test competition.

## Setup (Instructor)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate the workshop database
python generate_data.py

# Start the server
python app.py
```

The server runs on `http://localhost:8000`. Share this URL (or your machine's IP) with students.

## Workshop Flow

| Phase | Time | Activity | Tool |
|-------|------|----------|------|
| 1. Explore | 45 min | Query the database, discover user segments and funnel dynamics | SQL Explorer (`/sql`) |
| 2. Build | 75 min | Feature engineering + model training in Jupyter | `notebooks/student_workbook.ipynb` |
| 3. Optimize | 30 min | Profile latency, navigate accuracy-latency tradeoff | Notebook |
| 4. Compete | 45 min | Upload models, run live A/B test, watch dashboard | Dashboard (`/dashboard`) |

## Key Learning Moments

1. **CTR is not revenue**: "Window shopper" users click everything but never buy. "Researcher" users rarely click but make large purchases when they do. A pure CTR model shows clickbait to window shoppers — high clicks, no revenue.

2. **Full-funnel modeling**: The winning strategy models `P(click) * P(convert|click) * revenue`, not just `P(click)`.

3. **Clickbait hurts**: High clickbait scores boost CTR but *reduce* conversion rates. This is intentionally baked into the data generating process.

4. **Latency matters**: Complex models are penalized in the live simulation. If inference exceeds the latency budget, traffic gets randomly assigned instead of using the model's prediction.

## Student Workflow

1. Open `notebooks/student_workbook.ipynb`
2. Set `SERVER` to the instructor's URL
3. Register their team
4. Explore data via SQL (in notebook or at `/sql`)
5. Train and evaluate models
6. Wrap model in a `ScoringModel` class with a `.predict()` method
7. Pickle and upload to the server
8. Watch the live dashboard at `/dashboard`

## Architecture

```
generate_data.py    → Creates SQLite database with synthetic e-commerce data
app.py              → FastAPI server (SQL API, model upload, simulation control)
simulator.py        → A/B test engine (scores teams' models against ground-truth DGP)
static/             → Web UI (landing page, SQL explorer, live dashboard)
notebooks/          → Student Jupyter notebook
data/workshop.db    → Generated database (not in git)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sql` | Run read-only SQL query |
| GET | `/api/schema` | Database schema |
| POST | `/api/teams/{name}/register` | Register a team |
| POST | `/api/teams/{name}/model` | Upload a pickled model |
| GET | `/api/teams` | List teams |
| GET | `/api/leaderboard` | Current standings |
| POST | `/api/simulation/start` | Start A/B test (instructor) |
| POST | `/api/simulation/stop` | Stop A/B test |

## Deployment Options

**Local network**: Run on a laptop, students connect via IP address.

**Cloud**: Deploy to any platform that runs Python (Render, Railway, Fly.io, EC2). The SQLite database is self-contained.

## Data Generating Process

The synthetic data has five hidden user segments with distinct behavioral profiles:

| Segment | % | CTR | CVR\|click | Revenue/Impression |
|---------|---|-----|-----------|-------------------|
| Window Shoppers | 30% | High | Very low | Low |
| Bargain Hunters | 25% | Medium | Medium (discount-dependent) | Medium |
| Loyal Customers | 20% | Medium-low | High | High |
| Impulse Buyers | 15% | Medium | Medium | Medium |
| Researchers | 10% | Low | Very high | Highest |

Students don't see segment labels — they must discover these patterns through data exploration.
