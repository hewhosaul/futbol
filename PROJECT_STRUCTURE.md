# Mega Football Predictor - Project Structure

## File Organization

```
mega-football-predictor/
├── mega_football_predictor.py    # Main script (89KB, 2381 lines)
├── README.md                      # Comprehensive documentation
├── QUICKSTART.md                  # Quick start guide
├── PROJECT_STRUCTURE.md           # This file
├── requirements.txt               # Python dependencies
├── example_config.json            # Sample configuration
├── .gitignore                     # Git ignore rules
│
├── data/                          # Input data directory
│   ├── matches.csv               # Your match data (optional)
│   ├── statsbomb/                # StatsBomb data (optional)
│   └── fbref/                    # FBref data (optional)
│
├── outputs/                       # Generated outputs
│   ├── master_dashboard.png      # Main visualization (317KB)
│   ├── predictions.csv           # Model predictions
│   ├── trades.csv                # Betting simulation results
│   ├── features.csv              # Engineered features
│   ├── run_log.txt               # Detailed execution log
│   └── summary_report.txt        # Summary metrics
│
└── cache/                        # Cached computations
    └── *.pkl                     # Pickled cache files

```

## Script Architecture (mega_football_predictor.py)

### Header (Lines 1-150)
- Module docstring with installation instructions
- Imports (numpy, pandas, sklearn, torch, etc.)
- Optional dependency handling (HAS_XGB, HAS_TORCH, etc.)
- Matplotlib backend configuration

### Configuration (Lines 151-200)
- `Config` dataclass - Central configuration
- Global CONFIG instance
- Logging setup

### Utilities (Lines 201-250)
- `hash_dict()` - Cache key generation
- `cache_to_disk()` - Decorator for disk caching
- `safe_divide()` - Safe division
- `calculate_haversine_distance()` - Geographic distance

### Data Models (Lines 251-350)
- `Match` dataclass - Represents a single match
  - Basic info (teams, scores, date, league)
  - Market odds (home/draw/away)
  - Optional data (lineups, weather, events, tracking)
  - `result()` method returns H/D/A

### Data Loading (Lines 351-600)
- `DataLoader` class
  - `load_all_data()` - Main entry point
  - `_load_statsbomb_data()` - StatsBomb connector
  - `_load_fbref_data()` - FBref connector
  - `_load_local_csv_data()` - Local CSV loader
  - `_generate_synthetic_data()` - Synthetic data generator
  - `_enrich_with_odds()` - Odds API connector
  - `_enrich_with_weather()` - Weather API connector
  - `_enrich_with_referee_stats()` - Referee data

### Feature Engineering (Lines 601-950)
- `FeatureEngineer` class
  - `engineer_features()` - Main pipeline
  - `_engineer_match_features()` - Per-match features (120+)
    - Match-level: home advantage, travel, rest days
    - Team-level: rolling stats, season aggregates, form
    - Elo ratings: dynamic team strength
    - Head-to-head: historical matchup stats
    - Market-based: implied probabilities, margins
    - Derived: attack vs defense matchups
  - Helper methods:
    - `_get_days_since_last_match()`
    - `_count_recent_matches()`
    - `_calculate_travel_distance()`
    - `_calculate_team_stats()`
    - `_calculate_h2h_stats()`
    - `_update_elo_ratings()`

### Models (Lines 951-1650)

#### Base Class (Lines 951-1012)
- `BaseModel` - Abstract base class
  - `fit()` - Train model
  - `predict()` - Make predictions
  - `predict_proba()` - Probability predictions
  - `_prepare_features()` - Feature preprocessing

#### Statistical Models (Lines 1013-1100)
- `PoissonModel` - Bivariate Poisson regression
  - Separate models for home/away goals
  - Scoreline probability distributions
  - Outcome probability calculation

#### Machine Learning Models (Lines 1101-1400)
- `XGBoostModel` - Gradient boosting (GPU support)
- `LightGBMModel` - Fast gradient boosting
- `RandomForestModel` - Ensemble baseline

#### Deep Learning Models (Lines 1401-1550)
- `LSTMModel` - Sequence model for time-series
  - `LSTMNet` - PyTorch neural network
  - `_create_sequences()` - Sequence preparation
  - GPU/CPU automatic detection

#### Regime Models (Lines 1551-1620)
- `HMMRegimeModel` - Hidden Markov Model
  - Gaussian HMM for regime detection
  - Regime-conditioned predictions

#### Ensemble (Lines 1621-1650)
- `EnsembleModel` - Meta-learner stacking
  - Combines all base models
  - Logistic regression meta-model
  - Fallback to simple averaging

### Backtesting (Lines 1651-1800)
- `BacktestEngine` class
  - `run_backtest()` - Walk-forward validation
  - Train/test split by date
  - Model training loop
  - Prediction generation

### Betting Evaluation (Lines 1801-1950)
- `BettingEvaluator` class
  - `evaluate_betting_strategy()` - Main evaluation
  - `_simulate_betting()` - Per-model simulation
    - Expected value calculation
    - Kelly criterion stake sizing
    - P&L calculation
    - Performance metrics (ROI, win rate)

### Metrics (Lines 1951-2050)
- `MetricsCalculator` class
  - `calculate_all_metrics()` - Classification metrics
    - Accuracy
    - Log loss
    - Brier score
    - Confusion matrix
  - `calculate_calibration()` - Calibration curves

### Visualization (Lines 2051-2250)
- `DashboardGenerator` class
  - `generate_master_dashboard()` - Main dashboard (6x3 grid)
    - Model performance comparison
    - Equity curves
    - Calibration plots (H/D/A)
    - Confusion matrices (top 3 models)
    - Feature importance placeholder
    - Metrics summary table
  - Plotting methods:
    - `_plot_model_comparison()`
    - `_plot_equity_curves()`
    - `_plot_calibration()`
    - `_plot_confusion_matrix()`
    - `_plot_feature_importance()`
    - `_plot_metrics_table()`

### Main Orchestrator (Lines 2251-2380)
- `MegaFootballPredictor` class
  - `run()` - Main execution pipeline
    1. Load data
    2. Engineer features
    3. Initialize models
    4. Run backtesting
    5. Evaluate betting
    6. Calculate metrics
    7. Generate dashboard
    8. Generate summary report
  - `_initialize_models()` - Create model instances
  - `_generate_summary_report()` - Final report

### Entry Point (Lines 2381+)
- `parse_arguments()` - CLI argument parsing
- `main()` - Entry point
  - Config initialization
  - Command-line handling
  - Predictor execution

## Key Design Decisions

### Single-File Architecture
**Why?** Simplicity, portability, easier distribution
- All code in one file (89KB)
- No package installation needed
- Can be run directly: `python mega_football_predictor.py`

### Defensive Programming
**Why?** Never crash on missing data
- Try/except around all external I/O
- Graceful degradation (skip missing models)
- Comprehensive logging
- Fallbacks for all data sources

### Synthetic Data Generation
**Why?** Zero-dependency demo mode
- Works out-of-the-box without any data
- Realistic match simulation
- Perfect for testing and CI/CD

### GPU with CPU Fallback
**Why?** Performance + portability
- PyTorch CUDA detection
- XGBoost GPU support
- Automatic fallback to CPU
- No crashes on GPU-less systems

### Walk-Forward Backtesting
**Why?** No lookahead bias
- Realistic temporal validation
- Models retrained periodically
- Mimics production deployment

### Ensemble Meta-Learning
**Why?** Best of all models
- Stacking with logistic regression
- Learns optimal model weights
- Fallback to averaging

## Data Flow

```
Raw Data (CSV/API/Synthetic)
    ↓
DataLoader
    ↓
List[Match] objects
    ↓
FeatureEngineer
    ↓
DataFrame (120+ features)
    ↓
Train/Test Split
    ↓
Models (fit on train, predict on test)
    ↓
Predictions DataFrame
    ↓
├─→ BettingEvaluator → Trades DataFrame
├─→ MetricsCalculator → Metrics
└─→ DashboardGenerator → master_dashboard.png
    ↓
Output Files (CSV, PNG, TXT)
```

## Configuration Options

### Execution Modes
- `test_mode`: Run with 5 synthetic matches (~30s)
- `fast_mode`: Use fewer models, less data
- `use_gpu`: Enable GPU acceleration
- `n_jobs`: Parallel CPU cores (-1 = all)

### Data Sources
- `use_tracking_data`: Spatial/positional data
- `use_odds_api`: Live odds from APIs
- `use_sentiment`: Social media sentiment
- `use_weather`: Weather API integration

### Modeling
- `models_to_train`: List of models to use
- `max_model_trials`: Hyperparameter search limit
- `retrain_frequency`: Matches between retrains

### Backtesting
- `train_test_split_date`: Split point (default: 2023-01-01)
- `walk_forward_window`: Training window size
- `walk_forward_step`: Step size for rolling window

### Betting
- `min_edge_threshold`: Minimum edge to bet (default: 0.05)
- `max_bet_fraction`: Max fraction of bankroll (default: 0.05)
- `kelly_fraction`: Fractional Kelly (default: 0.25)

## Output Files Explained

### master_dashboard.png (317KB)
6-row by 3-column grid showing:
1. **Row 1**: Model performance comparison (accuracy, log loss, Brier)
2. **Row 2**: Equity curves (cumulative P&L over time)
3. **Row 3**: Calibration plots for H/D/A outcomes
4. **Row 4**: Confusion matrices (top 3 models)
5. **Row 5**: Feature importance (placeholder)
6. **Row 6**: Performance metrics table

### predictions.csv
One row per test match with columns:
- Match info (id, date, teams, actual result, scores, odds)
- Per-model probabilities (`[Model]_prob_H/D/A`)
- Per-model predictions (`[Model]_pred`)

### trades.csv
One row per bet with columns:
- Match info
- `bet_outcome`: H/D/A
- `stake`: Amount bet (Kelly criterion)
- `odds`: Market odds taken
- `prob`: Model's probability
- `edge`: Estimated edge
- `won`: Boolean
- `pnl`: Profit/loss
- `model`: Which model made the bet

### features.csv
One row per match with 120+ feature columns:
- Match context (rest days, travel, weather)
- Team stats (goals, xG, form, etc.)
- Elo ratings
- H2H stats
- Market features (odds, implied probs)
- Derived features

### run_log.txt (200KB+)
Detailed execution log with:
- INFO: Normal operation messages
- WARNING: Non-fatal issues (missing data, model failures)
- ERROR: Failures (with graceful handling)
- Timestamps for all events

### summary_report.txt
Concise text summary:
- Best model
- Key metrics (accuracy, Brier, log loss)
- Betting performance (ROI, win rate, return)
- Output file paths

## Extending the System

### Add a Data Source
1. Add method to `DataLoader`: `_load_my_source()`
2. Return `List[Match]`
3. Call from `load_all_data()`

### Add a Model
1. Extend `BaseModel` class
2. Implement `fit()` and `predict_proba()`
3. Add to `_initialize_models()` in orchestrator
4. Add name to `models_to_train` in config

### Add Features
1. Edit `_engineer_match_features()` in `FeatureEngineer`
2. Add feature calculation logic
3. Add to `features` dict
4. Will automatically be used by all models

### Add Visualization
1. Edit `generate_master_dashboard()` in `DashboardGenerator`
2. Add subplot to grid
3. Implement plotting method
4. Call from main dashboard generation

## Performance Characteristics

### Speed (Test Mode)
- Data loading: <1s
- Feature engineering: 1-2s
- Model training: 10-20s
- Predictions: <1s
- Dashboard generation: 5s
- **Total: ~30s**

### Speed (Full Run, 500 matches)
- Data loading: 1-5s
- Feature engineering: 10-30s
- Model training: 2-10 minutes
- Predictions: 1-5s
- Dashboard generation: 10-20s
- **Total: 3-15 minutes**

### Memory Usage
- Test mode: <500MB
- Full run: 1-3GB
- GPU models: +2-4GB VRAM

### Accuracy (Typical)
- **Accuracy**: 50-60% (vs 33% random)
- **Log Loss**: 0.9-1.2 (lower is better)
- **Brier Score**: 0.20-0.25 (lower is better)
- **ROI**: -5% to +10% (highly variable)

## Troubleshooting Guide

### "No module named 'xyz'"
→ Install: `pip install xyz`

### "CUDA not available"
→ Normal on CPU-only systems, models use CPU fallback

### "Failed to train [Model]"
→ Normal if optional dependency missing, system continues

### Low accuracy in test mode
→ Expected with only 5 matches, use real data for meaningful results

### Negative ROI
→ Normal with small samples or synthetic data, tune `min_edge_threshold`

### "Scaler fit failed"
→ Handled automatically with fallback to unscaled features

### Empty predictions
→ Check that test data has future dates after `train_test_split_date`

## Best Practices

1. **Start with test mode** - Verify installation works
2. **Use real data** - Synthetic data is for demo only
3. **Tune edge threshold** - Start conservative (10%+)
4. **Monitor calibration** - Well-calibrated = reliable
5. **Backtest thoroughly** - 100+ matches minimum
6. **Update regularly** - Retrain weekly with new data
7. **Fractional Kelly** - Use 0.1-0.25 for safety
8. **Validate externally** - Compare with market consensus

## License & Disclaimer

**MIT License** - Free to use, modify, distribute

**Educational Use Only** - Not financial advice, sports betting involves risk

---

For questions, see README.md or check run_log.txt for detailed execution logs.
