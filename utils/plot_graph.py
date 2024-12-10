import numpy as np
import matplotlib.pyplot as plt

# Plot graphs from raw data for the final report

linear = np.array([56.34997295930816, 50.97222587927723, 55.597483754031316, 49.85173601135608])
cosine = np.array([179.34364456507058, 96.14443109638893, 52.07691197696556, 43.48567534748355])
sd = np.array([60.25653959236513, 58.015990520053094, 70.26783066523944, 52.55208185476181])
epochs = np.array([50, 100, 150, 200])
font_size = 15

# File_data = np.loadtxt("train_stats/cosine_200_epoch_loss.txt", dtype=float)[0:200]
plt.figure()
plt.plot(epochs, linear, label='linear')
plt.scatter(epochs, linear)
plt.plot(epochs, cosine, label='cosine')
plt.scatter(epochs, cosine)
plt.plot(epochs, sd, label='sd')
plt.scatter(epochs, sd)
plt.xlabel('Epochs', fontsize=font_size)
plt.ylabel('FID Score', fontsize=font_size)
plt.legend(fontsize=font_size)
plt.title('FID Score  - DDPM', fontsize=font_size)
plt.savefig('data_report/ddpm.png')

