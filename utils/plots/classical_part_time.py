import matplotlib.pyplot as plt
import sys
sys.path.insert(1, './extract_data')
from extract_data_from_classical_part import extract_data


N = [15, 21, 33, 35, 39, 51, 55, 57]
time_ceil_ceil = extract_data("ceil_ceil", 1, len(N))[-1]
time_ceil_floor = extract_data("ceil_floor", 1, len(N))[-1]
time_floor_ceil = extract_data("floor_ceil", 1, len(N))[-1]
time_floor_floor = extract_data("floor_floor", 1, len(N))[-1]

time_ceil_ceil = [i / 1000 for i in time_ceil_ceil]
time_ceil_floor = [i / 1000 for i in time_ceil_floor]
time_floor_ceil = [i / 1000 for i in time_floor_ceil]
time_floor_floor = [i / 1000 for i in time_floor_floor]

# odejmowanie części klasycznej ze względu na zaokrąglenie do ms nie ma sensu
time_shor = [0.03266334533691406, 0.026106834411621094, 0.018596649169921875, 0.010967254638671875,
             0.017881393432617188, 0.02130866050720215, 0.0162187375520405, 0.014046827952067057]

plt.xlabel("N - factorized number")
plt.ylabel("time [h]")
plt.plot(N[:len(time_ceil_ceil)], time_ceil_ceil, label="Regev's algorithm ceil_ceil")
plt.plot(N[:len(time_ceil_floor)], time_ceil_floor, label="Regev's algorithm ceil_floor")
plt.plot(N[:len(time_floor_ceil)], time_floor_ceil, label="Regev's algorithm floor_ceil")
plt.plot(N[:len(time_floor_floor)], time_floor_floor, label="Regev's algorithm floor_floor")
plt.plot(N[:len(time_shor)], time_shor, label="Shor's algorithm")

plt.title("Time execution of classical part")
# plt.legend(bbox_to_anchor=(0, 0.92, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.subplots_adjust(bottom=0.35)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
plt.grid(color='gray', linestyle='--', linewidth=0.25)
plt.savefig("./../../images/plots/classical_part_time.png")
