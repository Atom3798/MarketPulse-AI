import matplotlib.pyplot as plt

class ReportGenerator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def plot(self, save_path="forecast.png"):
        plt.figure()
        plt.plot(self.y_true, label="Actual")
        plt.plot(self.y_pred, label="Predicted")
        plt.legend()
        plt.title("MarketPulse AI Forecast")
        plt.savefig(save_path)
