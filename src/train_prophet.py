
import pandas as pd
from prophet import Prophet

df=pd.read_csv('../data/synthetic.csv')
m=Prophet()
m.fit(df)
future=m.make_future_dataframe(90)
forecast=m.predict(future)
forecast.to_csv('../data/forecast.csv', index=False)
