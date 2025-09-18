from marketpulse.scraper import MarketScraper
from marketpulse.data_manager import DataManager
from marketpulse.model import MarketModel
from marketpulse.report import ReportGenerator

class MarketPulsePipeline:
    def __init__(self):
        self.scraper = MarketScraper()
        self.model = MarketModel()

    def run(self, ticker="AAPL"):
        raw = self.scraper.scrape_stock(ticker)
        df = DataManager(raw).engineer_features()
        X, y = df[["ma5", "volatility"]], df["price"]

        self.model.train(X, y)
        y_pred = self.model.predict(X)

        report = ReportGenerator(y, y_pred)
        report.plot(f"{ticker}_forecast.png")

        return {"ticker": ticker, "evaluation": self.model.evaluate(X, y)}
