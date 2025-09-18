from marketpulse.pipeline import MarketPulsePipeline
import json

pipe = MarketPulsePipeline()
result = pipe.run("AAPL")
print(json.dumps(result, indent=2))
