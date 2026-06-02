import matplotlib.pyplot as plt

class ReportGenerator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def plot(self, save_path="forecast.png"):
        fig, ax = plt.subplots()
        ax.plot(self.y_true.values, label="Actual")
        ax.plot(self.y_pred, label="Predicted")
        ax.legend()
        ax.set_title("MarketPulse AI Forecast")
        fig.savefig(save_path)
        plt.close(fig)
