import matplotlib.pyplot as plt
import numpy as np

# quantum part

N = [15, 21, 33, 35, 39, 51, 55, 57]
regev_ceil_ceil_time_ms = [8318, 1600148, 10606411, 11018810, 11147302, 10825613, 11377694, 11185650]
regev_ceil_ceil_h = [i // 3600000 for i in regev_ceil_ceil_time_ms]
regev_ceil_floor_time_ms = [15470, 286398, 15425238, 15425238, 11207907, 10886042, 15971179, 11183239]
regev_ceil_floor_h = [i // 3600000 for i in regev_ceil_floor_time_ms]
# odejmowanie części klasycznej ze względu na zaokrąglenie do ms nie ma sensu
shor_time_ms_all = [17820, 67915, 1083631, 1097304, 1118240, 1087438, 1151647, 1139126]
shor_h = [i // 3600000 for i in shor_time_ms_all]

plt.xlabel("N - factorized number")
plt.ylabel("time [h]")
plt.plot(N, regev_ceil_ceil_h, label="Regev's algorithm ceil_ceil")
plt.plot(N, regev_ceil_floor_h, label="Regev's algorithm ceil_floor")
plt.plot(N, shor_h, label="Shor's algorithm")
plt.title("Quantum computations executing time")
# plt.legend(bbox_to_anchor=(0, 0.92, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.subplots_adjust(bottom=0.25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
plt.grid(color='gray', linestyle='--', linewidth=0.25)
plt.savefig("./../../images/plots/regev_ceil_quantum_time.png")
