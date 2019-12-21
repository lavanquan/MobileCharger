import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("energy_per_second.csv")
plt.plot(df["Time"], df["Greedy"])
plt.show()