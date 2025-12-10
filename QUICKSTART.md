# Quick Start Guide - Mega Football Predictor

## Installation (5 minutes)

### Step 1: Install Python Dependencies

```bash
# Basic installation (CPU only)
pip install -r requirements.txt

# Or install with --break-system-packages if needed
pip install --break-system-packages -r requirements.txt
```

### Step 2: Quick Test Run

```bash
# Run a quick test with synthetic data (completes in ~30 seconds)
python mega_football_predictor.py --test
```

This will:
- Generate 5 synthetic matches
- Train 3 models (Poisson, RandomForest, Ensemble)
- Create predictions and backtest results
- Generate a master dashboard at `outputs/master_dashboard.png`

## Understanding the Outputs

After running, check the `outputs/` directory:

### 1. Master Dashboard (`master_dashboard.png`)
- Visual summary of all results
- Model performance comparison
- Equity curves for betting strategies
- Calibration plots
- Confusion matrices
- Metrics table

### 2. Predictions (`predictions.csv`)
Columns include:
- `match_id`, `date`, `home_team`, `away_team`
- `actual_result` - H/D/A (Home/Draw/Away)
- `[Model]_prob_H/D/A` - Predicted probabilities for each outcome
- `[Model]_pred` - Predicted result
- `odds_home/draw/away` - Market odds

### 3. Trades (`trades.csv`)
Simulated betting performance:
- `bet_outcome` - What was bet on (H/D/A)
- `stake` - Amount staked (Kelly criterion)
- `odds` - Odds taken
- `edge` - Estimated edge over bookmaker
- `won` - Whether bet won
- `pnl` - Profit/loss for the bet

### 4. Features (`features.csv`)
All 120+ engineered features per match

### 5. Summary Report (`summary_report.txt`)
Concise text summary with best model and key metrics

## Using Your Own Data

### Option 1: CSV File (Recommended for beginners)

Create `data/matches.csv` with these columns:

```csv
match_id,date,league,season,home_team,away_team,home_score,away_score,home_xg,away_xg,odds_home,odds_draw,odds_away
M001,2023-08-12,Premier League,2023-2024,Arsenal,Chelsea,2,1,1.8,1.2,2.10,3.40,3.80
M002,2023-08-13,Premier League,2023-2024,Liverpool,Man United,3,0,2.5,0.8,1.85,3.50,4.20
```

**Required columns:**
- `match_id` - Unique identifier
- `date` - Match date (YYYY-MM-DD)
- `home_team`, `away_team` - Team names
- `home_score`, `away_score` - Goals scored
- `league`, `season` - For organizing data

**Optional columns (improves predictions):**
- `home_xg`, `away_xg` - Expected goals
- `odds_home`, `odds_draw`, `odds_away` - Bookmaker odds (decimal format)

Then run:
```bash
python mega_football_predictor.py
```

### Option 2: Advanced Data Sources

For StatsBomb, FBref, or other sources, place data in:
- `data/statsbomb/` - StatsBomb JSON files
- `data/fbref/` - FBref CSV exports
- Configure API keys in config for live data

## Common Use Cases

### 1. Testing a New Strategy
```bash
# Edit example_config.json to set:
# - "min_edge_threshold": 0.10  (only bet with 10%+ edge)
# - "kelly_fraction": 0.1  (bet 10% of Kelly for safety)

python mega_football_predictor.py --config example_config.json
```

### 2. Quick Experimentation
```bash
# Fast mode: fewer models, quicker results
python mega_football_predictor.py --fast
```

### 3. CPU-Only Mode
```bash
# If no GPU or to avoid GPU errors
python mega_football_predictor.py --no-gpu
```

### 4. Production Backtesting
```bash
# Full run with all models (may take 10-30 minutes depending on data size)
python mega_football_predictor.py
```

## Interpreting Results

### Model Metrics

**Accuracy** (0-1, higher better)
- Percentage of correct predictions
- 0.33 = random guessing (3 outcomes)
- 0.50+ = decent performance
- 0.55+ = strong performance

**Log Loss** (0-∞, lower better)
- Measures quality of probability predictions
- <1.0 = good
- <0.8 = excellent

**Brier Score** (0-1, lower better)
- Accuracy of probability forecasts
- 0.25 = random
- <0.22 = good
- <0.20 = excellent

### Betting Metrics

**ROI** (Return on Investment)
- Percentage return on all bets
- Positive = profitable
- 5%+ = very good
- 10%+ = exceptional (rarely sustained)

**Win Rate**
- Percentage of winning bets
- Depends on odds taken
- Not meaningful alone (can be profitable with 30% win rate if odds are high)

**Edge**
- Estimated advantage over bookmaker
- System only bets when edge > threshold
- Higher edge = more confident bet

## Troubleshooting

### Error: "No module named 'xyz'"
```bash
pip install xyz
# or
pip install --break-system-packages xyz
```

### Warning: "Failed to train [Model]"
- Normal if optional dependencies missing
- System continues with available models
- Install missing packages for full functionality

### Low Accuracy in Test Mode
- Test mode uses only 5 matches - too small for meaningful results
- Run without `--test` flag and provide real data

### Negative ROI
- Normal with small sample sizes or synthetic data
- Ensure sufficient historical data (100+ matches)
- Tune `min_edge_threshold` higher (e.g., 0.10)
- Use fractional Kelly (0.1-0.25) for safety

## Next Steps

1. **Add Real Data**: Replace synthetic data with actual match results
2. **Tune Parameters**: Adjust `example_config.json` for your use case
3. **Monitor Calibration**: Check calibration plots - well-calibrated = reliable
4. **Backtest Thoroughly**: Test on historical data before any real betting
5. **Update Regularly**: Retrain models weekly with new match data

## Advanced Features

### Custom Feature Engineering
Edit `FeatureEngineer._engineer_match_features()` in the script to add domain-specific features.

### Add New Models
Extend `BaseModel` class and add to `models_to_train` in config.

### Live Predictions
Provide upcoming matches (without scores) in CSV, and models will predict outcomes.

### API Integration
Set API keys in config for live odds, weather, and sentiment data.

## Important Disclaimers

⚠️ **For Educational Use Only**
- This is a research/educational tool
- Not financial advice
- Sports betting involves risk of loss
- Past performance ≠ future results

⚠️ **Responsible Usage**
- Only bet what you can afford to lose
- Set strict bankroll limits
- Validate predictions extensively before trusting
- Consider this experimental software

## Support

- Check `outputs/run_log.txt` for detailed execution logs
- Review `README.md` for comprehensive documentation
- Test with `--test` flag first to verify installation

## Example Workflow

```bash
# 1. Quick test
python mega_football_predictor.py --test

# 2. View dashboard
open outputs/master_dashboard.png

# 3. Add your data to data/matches.csv

# 4. Run full backtest
python mega_football_predictor.py

# 5. Analyze results in outputs/

# 6. Tune configuration
nano example_config.json

# 7. Re-run with custom config
python mega_football_predictor.py --config example_config.json
```

---

**Ready to start?** Run `python mega_football_predictor.py --test` now! ⚽
