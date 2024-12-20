import matplotlib.pyplot as plt
import sys
sys.path.insert(1, './extract_data')
from extract_data_from_classical_part import extract_data

N = [15, 21, 33, 35, 39, 51, 55, 57]

regev_ceil_ceil_time_ms = [8318, 1600148, 10606411, 11018810, 11147302, 10825613, 11377694, 11185650]
# regev_ceil_ceil_h = [i // 3600000 for i in regev_ceil_ceil_time_ms]
regev_ceil_ceil_s = [i / 1000 for i in regev_ceil_ceil_time_ms]
regev_ceil_floor_time_ms = [15470, 286398, 15425238, 15425238, 11207907, 10886042, 15971179, 11183239]
# regev_ceil_floor_h = [i // 3600000 for i in regev_ceil_floor_time_ms]
regev_ceil_floor_s = [i / 1000 for i in regev_ceil_floor_time_ms]
regev_floor_ceil_time_ms = [8542, 47161, 245911, 256074, 257180, 251896, 262070, 257281]
# regev_floor_ceil_h = [i // 3600000 for i in regev_floor_ceil_time_ms]
regev_floor_ceil_s = [i / 1000 for i in regev_floor_ceil_time_ms]
regev_floor_floor_time_ms = [14698, 34719, 383576, 399456, 405097, 364752, 446004, 369373]
# regev_floor_floor_h = [i // 3600000 for i in regev_floor_floor_time_ms]
regev_floor_floor_s = [i / 1000 for i in regev_floor_floor_time_ms]
shor_time_ms_all = [17820, 67915, 1083631, 1097304, 1118240, 1087438, 1151647, 1139126]
shor_s = [i / 1000 for i in shor_time_ms_all]


factorize_effectiveness_ceil_ceil = extract_data("ceil_ceil", 1, len(N))[2]
# factorize_effectiveness_ceil_ceil = [i / 100 for i in factorize_effectiveness_ceil_ceil]
factorize_effectiveness_ceil_floor = extract_data("ceil_floor", 1, len(N))[2]
# factorize_effectiveness_ceil_floor = [i / 100 for i in factorize_effectiveness_ceil_floor]
factorize_effectiveness_floor_ceil = extract_data("floor_ceil", 1, len(N))[2]
# factorize_effectiveness_floor_ceil = [i / 100 for i in factorize_effectiveness_floor_ceil]
factorize_effectiveness_floor_floor = extract_data("floor_floor", 1, len(N))[2]
# factorize_effectiveness_floor_floor = [i / 100 for i in factorize_effectiveness_floor_floor]
shor_effectiveness = [1/2, 1/2, 18/20, 3/4, 18/20, 14/16, 33/38, 21/24]
shor_effectiveness = [i * 100 for i in shor_effectiveness]

m_regev_ceil_ceil = [t / e for t, e in zip(regev_ceil_ceil_s, factorize_effectiveness_ceil_ceil)]
m_regev_ceil_floor = [t / e for t, e in zip(regev_ceil_floor_s, factorize_effectiveness_ceil_floor)]
m_regev_floor_ceil = [t / e for t, e in zip(regev_floor_ceil_s, factorize_effectiveness_floor_ceil)]
m_regev_floor_floor = [t / e for t, e in zip(regev_floor_floor_s, factorize_effectiveness_floor_floor)]
m_shor = [t / e for t, e in zip(shor_s, shor_effectiveness)]
print(m_regev_ceil_ceil)
print(m_regev_ceil_floor)
print(m_regev_floor_ceil, regev_floor_ceil_s, factorize_effectiveness_floor_ceil)
print(m_regev_floor_floor, regev_floor_floor_s, factorize_effectiveness_floor_floor)
plt.xlabel("N - factorized number")
plt.ylabel("effectiveness / quantum time execution [% / s]")
plt.plot(N[:len(N)], m_regev_ceil_ceil, label="Regev's algorithm ceil_ceil")
plt.plot(N[:len(N)], m_regev_floor_ceil, label="Regev's algorithm ceil_floor")
plt.plot(N[:len(N)], m_regev_ceil_floor, label="Regev's algorithm floor_ceil")
plt.plot(N[:len(N)], m_regev_floor_floor, label="Regev's algorithm floor_floor")
plt.plot(N[:len(N)], m_shor, label="Shor's algorithm")

plt.title("Comparison of effectiveness-time metric")
# plt.legend(bbox_to_anchor=(0, 0.92, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.subplots_adjust(bottom=0.35)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
plt.grid(color='gray', linestyle='--', linewidth=0.25)
plt.savefig("./../../images/plots/time_divided_by_effectiveness.png")
