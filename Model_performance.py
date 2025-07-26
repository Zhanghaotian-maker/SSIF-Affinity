import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

file_path = './result/predicted_affinities_test.csv'
data = pd.read_csv(file_path, usecols=['Predicted_Affinity', 'True_Affinity'])  

predicted = data['Predicted_Affinity'].values
true = data['True_Affinity'].values

rmse = np.sqrt(np.mean((predicted - true) ** 2))
mae = np.mean(np.abs(predicted - true))
corr, _ = pearsonr(predicted, true)
r = corr

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R: {r}")

plt.figure(figsize=(10, 6))
plt.scatter(true, predicted, alpha=0.7, edgecolors='w', linewidths=0.5)
plt.plot([min(true), max(true)], [min(true), max(true)], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('True Affinity')
plt.ylabel('Predicted Affinity')
plt.title('True vs Predicted Affinity')
plt.grid(True)
plt.legend()
plt.show()

