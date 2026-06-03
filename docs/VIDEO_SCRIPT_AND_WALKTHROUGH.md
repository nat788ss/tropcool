# TropCool — Video script & walkthrough (≤10 min)

**Title:** TropCool — Transformer world model + RL for tropical data-center cooling  
**Track:** Application / Product *(research backbone in Q2)*  
**Target length:** **7–9 minutes** (hard cap 10 min)  
**Repo:** https://github.com/nat788ss/tropcool  

---

## Timing overview

| Section | Time | On screen |
|---------|------|-----------|
| **Q1** — Why | 1:30–2:00 | Talking head or title slide → map |
| **Q2** — How | 3:30–4:30 | Dashboard + optional terminal / How It Works tab |
| **Q3** — Use cases & impact | 1:30–2:00 | Explore results + Regional Impact |
| **Q4** — What more | 1:00–1:30 | Brief bullet slide or README |
| **Buffer / transitions** | ~0:30 | — |

---

## Before you record

```bash
cd /Users/nsivarajah/tropcool
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Open http://localhost:8501 → **⋮ → Rerun** (fresh JSON).

**Demo cities:** Singapore, Bangkok, or Ho Chi Minh for wins. **Jakarta:** show on map but say baseline-only.

Optional backup terminal:

```bash
python3 compare_policies.py --rl-per-city-dir data/rl/per_city_v2 \
  --action-box baseline --episode-len 512 --mean-over-scenarios \
  --exclude-cities jakarta --require-sac-beats-baseline-all
```

---

# Q1 — Why did you build what you did? (1:30–2:00)

### Bottlenecks to name

1. **Climate mismatch** — DC cooling systems and playbooks are tuned for temperate sites; Southeast Asia runs high wet-bulb hours, so fixed setpoints waste energy or risk thermal margin.
2. **Slow evaluation** — Stepping a full physics plant model for every RL trial is too slow to explore hourly policies across years of weather.
3. **Planning under warming** — Operators need forward-looking what-if (2030–2050), but live experiments on real plant are expensive and risky.
4. **Regional gap** — Fast-growing ASEAN markets (Malaysia, Singapore, Thailand, Vietnam, Indonesia) lack an open, reproducible toolchain tied to **local weather archives**.

### What inspired you

- News/policy on **grid strain** and DC moratoriums in hot, humid markets.
- Open weather APIs (Open-Meteo) + public DC/HVAC datasets proving data exists.
- World models in RL (learn dynamics, then train policies inside the model) as a way to **iterate fast** before touching hardware.

### Script (read aloud)

> I built **TropCool** because tropical data centers are growing faster than the cooling software stack designed for them.  
>  
> The first bottleneck is **physics and climate**: when humidity is high, evaporative and mechanical cooling trade-offs change every hour, but most facilities still run **fixed chiller and fan setpoints**.  
>  
> The second bottleneck is **experiment cost**. A detailed plant simulator is trustworthy but **too slow** to train reinforcement learning over twenty years of hourly weather across six cities.  
>  
> The third is **planning**: operators are asked what happens in **2030 or 2040**, but they rarely have a cheap way to test policies under warmer weather without risking live equipment.  
>  
> I was inspired by the idea of a **learned surrogate**—a transformer that predicts next-hour temperature, power, water, PUE, and cost from real weather and controls—so we can train an **RL agent** safely in software, then show results in a **dashboard** planners can actually use.

### Optional B-roll (15 s)

- **Explore** tab → map with six pins (no narration needed).

---

# Q2 — How exactly does the product work? (3:30–4:30)

**Your track:** **[2] Application / Product** — emphasize architecture, dashboard, deployment-shaped workflow.  
**Also cover [1] Research** — training data and world model — in the first half of Q2.

---

## Q2A — Research backbone (1:45–2:15)

### Script

> **Training data** starts with **real hourly weather** from Open-Meteo for six cities: Cyberjaya, Singapore, Jakarta, Bangkok, Johor Bahru, and Ho Chi Minh City—roughly twenty years per site.  
>  
> I built a **physics-style simulator** (`dc_simulator.py`) with chiller COP curves, towers, and CRAH fans, and used it in `generate_data.py` to create **millions of simulated hours** by randomizing controls and IT load.  
>  
> The **world model** is a PyTorch **transformer**: twenty-four hours of context plus **next-hour controls** predict rack inlet temperature, cooling power, water, PUE, and operating cost. I **fine-tuned** on public Kaggle DC/HVAC CSVs so the model isn’t only synthetic. On held-out synthetic episodes, **PUE R² is about 0.91–0.93**.  
>  
> That model is the **scientific core**—everything downstream assumes its predictions are a useful surrogate.

### On screen

- Tab: **How It Works** → mermaid diagram (weather → simulator → transformer → SAC → dashboard).
- Tab: **Model accuracy** → R² table (10–15 s).

---

## Q2B — Product architecture & “deployment” (1:45–2:15)

### Script

> **Product shape:** TropCool is a **local Streamlit dashboard** plus a Python pipeline—no cloud required for the demo. Precomputed JSON drives the UI so reviewers get **instant** what-if without retraining.  
>  
> **Architecture:**  
> - **Ingest:** weather CSVs and optional Kaggle real data.  
> - **Train:** world model checkpoint + **per-city SAC** policies in `data/rl/per_city_v2/`.  
> - **Evaluate:** `compare_policies.py` runs **baseline vs SAC** on the same twenty-seven weather scenarios per city—three years, three start times, three seeds, five-hundred-twelve-hour episodes.  
> - **Climate layer:** synthetic future weather files (+0.5 / +1.2 / +2.0 °C and optional **stress** +2.5 / +3.0 °C) fed through the same world model.  
> - **Present:** `dashboard/app.py` — map, facility sliders, horizon dropdown, headline cards, live twenty-four-hour animation, regional rollup.  
>  
> **Agent (SAC):** Soft Actor-Critic from Stable-Baselines3 acts in `WorldModelEnv`. Reward is **negative predicted cost** with a hard penalty if rack inlet exceeds **27 °C**. Training uses an **A+B recipe**: evaluation-aligned weather windows, baseline action box, five-hundred-twelve-hour episodes—so train and test contracts match.

### On screen — walkthrough (follow this order)

1. **Explore** — select **Singapore**, **20 MW**, water-cooled, horizon **2025** → Results metrics (PUE, annual cost, savings Y).
2. **Explore** — horizon **2040**:
   - **Blue card (IPCC +1.2 °C):** read **“SAC savings at that weather”** (~**6.8%** for Singapore — use your card).
   - **Orange card (stress +2.5 °C):** say it’s a **labeled stress scenario**, not the central IPCC path.
3. **Live Simulation** — **Ho Chi Minh** or **Singapore** → **Play** → mention end-of-day cumulative savings (10–20 s of animation).
4. *(Optional 20 s)* Terminal: strict eval table — **five cities pass**, Jakarta fails.

### Evaluation beat (inside Q2 — 30 s, builds rubric points)

> I don’t only show wins. **Jakarta** is excluded from our **validated five-city portfolio** because SAC still loses to baseline on the strict contract. We also ran a **simulator spot-check** on Singapore: SAC wins on the **world model** but **not always** on the physics simulator—so we treat the transformer as a **planning surrogate**, not a replacement for meters.

---

# Q3 — Potential use cases, impact, how people use it (1:30–2:00)

### Script

> **Use case 1 — Operator what-if:** Pick a city and facility size, compare **fixed baseline** vs **TropCool SAC** on predicted PUE, cost, and water before changing setpoints.  
>  
> **Use case 2 — Capacity planning:** Slide the **time horizon** to 2030–2050 and read **synthetic climate** headlines—how much baseline cost rises vs how much SAC saves **on that future weather**.  
>  
> **Use case 3 — Training & sales engineering:** The live **twenty-four-hour animation** explains *why* humidity-heavy hours need different fan/chiller behavior—even if every hour doesn’t look cheaper, **cumulative** cost can be.  
>  
> **Societal value:** Lower cooling energy and water in the hottest markets reduces **grid pressure** and emissions tied to new DC build-out—relevant where governments have paused approvals over power.  
>  
> **Who uses it:** Hyperscale planners, colo operators, and researchers benchmarking tropical policies—today as an **open-source prototype**; tomorrow as an API in front of BMS/DCIM.

### On screen

- **Regional Impact** tab — MW / water / CO₂ headlines (say **illustrative**, 10 MW reference per site).
- Return to **Explore** — Bangkok or Singapore **Results** bar chart.

---

# Q4 — What more would you add? (1:00–1:30)

### Script

> **Short term:** Fix **Jakarta** SAC with longer training or city-specific rewards; refresh `results.json` after strict eval passes six-for-six or keep honest five-city portfolio.  
>  
> **Science:** Replace **synthetic warming** with **CMIP6 or Open-Meteo climate API** downscaling; close the **simulator vs world-model gap** with periodic calibration.  
>  
> **Product:** Distill SAC into **interpretable rules** for BMS; add **MPC** and safety guardrails; deploy as a service behind customer weather feeds.  
>  
> **Governance:** Field trials with **metered validation**, rack-temperature alarms, and audit logs before any autonomous control.  
>  
> **Open source:** Push full commit history, containerize `streamlit run dashboard/app.py`, and document reproduce commands in the README.

### On screen

- GitHub README in browser, or static slide with four bullets.

---

## Closing line (10 s)

> TropCool is **real weather, a learned tropical surrogate, and per-city RL**—with honest evaluation and an interactive dashboard. Code and logs are on GitHub. Thank you.

---

## Numbers cheat sheet (read from dashboard if unsure)

| Claim | Safe line |
|-------|-----------|
| World model | PUE **R² ~ 0.92** |
| Strict eval | **5 cities pass** (exclude Jakarta from portfolio claims) |
| Singapore multi-scenario | **~7%** lower predicted cost vs baseline |
| 2040 weather (Singapore) | **~6.8%** SAC savings at same weather (`savings_at_weather_pct`) |
| Climate X/Y growth | Small % (fractions of a percent)—**don’t oversell**; lead with savings-at-weather |
| Jakarta | **Baseline recommended** |
| Climate data | **Synthetic ΔT** on historical years + optional **stress** scenario |

---

## Rubric alignment (quick)

| Criterion | Where in video |
|-----------|----------------|
| Problem & insight | Q1 bottlenecks + inspiration |
| Execution | Q2 pipeline + live dashboard |
| Evaluation | Q2 strict eval, Jakarta, simulator caveat, R² |
| Communication | Q1–Q4 structure, clear demo path |
| Integrity | Synthetic climate label, AI disclosure below, Jakarta honesty |

### AI & sources disclosure (say or put in description)

> I used **Cursor / Claude** for implementation help and documentation. I ran training, evaluation, and dashboard generation myself. **Data:** Open-Meteo, Kaggle DC/HVAC sets. **Libraries:** PyTorch, Stable-Baselines3, Gymnasium, Streamlit.

---

## Gradescope one-liners

**Progress:** End-to-end TropCool: simulator, transformer (PUE R² ~0.92), per-city SAC, 27-scenario eval (5-city validated portfolio), synthetic + stress climate, Streamlit dashboard.

**Future:** CMIP6 weather, Jakarta SAC fix, surrogate–sim calibration, rules distillation, production API.

**Track:** Application / Product

---

## Recording checklist

- [ ] GitHub pushed  
- [ ] Streamlit running, cache rerun  
- [ ] Mic test, 1080p, quiet room  
- [ ] Q1 → Q2 (research then product) → Q3 → Q4 in order  
- [ ] Under **10:00** — cut Live Sim to ~15 s if long  

Good take: one continuous screen recording of the dashboard with voiceover beats above.
