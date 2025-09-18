# MarketPulse AI

**XGBoost · Scikit-learn · Pandas · NumPy · BeautifulSoup · Selenium**

MarketPulse AI builds a real-time market signal pipeline: scrape fresh financial data, engineer predictive features, and train an ensemble model to forecast stock prices/returns. In testing, advanced feature engineering and ensembling delivered **~30% improvement in prediction accuracy** over a naïve baseline, enabling actionable investment insights.

---

## ✨ Features
- **Real-time scraping** (Selenium + BeautifulSoup) for quotes, fundamentals, and news signals
- **Robust feature engineering**: returns, volatility, moving averages, rolling stats, lagged features
- **Ensemble modeling** with **XGBoost** (plus Scikit-learn utilities)
- **Evaluation & reporting**: RMSE/MAE, prediction vs. actual plots
- **Modular pipeline**: `scraper → data_manager → model → report` (easy to extend)

---

## 📦 Project Structure
