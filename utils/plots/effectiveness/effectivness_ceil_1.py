import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../extract_data')
from extract_data_from_classical_part import extract_data

# quantum part

N, regev_ceil_ceil_effectiveness_1_N_1, regev_ceil_ceil_effectiveness_1_p_q, time = extract_data("ceil_ceil", 1, None)
N2, regev_ceil_floor_effectiveness_1_N_1, regev_ceil_floor_effectiveness_1_p_q, time2 = extract_data("ceil_floor", 1, None)


# odejmowanie części klasycznej ze względu na zaokrąglenie do ms nie ma sensu
shor_effectiveness= [1/2, 1/2, 18/20, 3/4, 18/20, 14/16, 33/38, 21/24]
shor_effectiveness = [100 * i for i in shor_effectiveness]

plt.xlabel("N - factorized number")
plt.ylabel("effectiveness [%]")
plt.plot(N[:len(regev_ceil_ceil_effectiveness_1_N_1)], regev_ceil_ceil_effectiveness_1_N_1,
         label="Regev's algorithm ceil_ceil - square root of unity modulo N", color="red")
plt.plot(N[:len(regev_ceil_ceil_effectiveness_1_p_q)], regev_ceil_ceil_effectiveness_1_p_q,
         label="Regev's algorithm ceil_ceil - non trivial square root of unity modulo N", color="orange")
plt.plot(N[:len(regev_ceil_floor_effectiveness_1_N_1)], regev_ceil_floor_effectiveness_1_N_1,
         label="Regev's algorithm ceil_floor - square root of unity modulo N", color="blue")
plt.plot(N[:len(regev_ceil_floor_effectiveness_1_p_q)], regev_ceil_floor_effectiveness_1_p_q,
         label="Regev's algorithm ceil_floor - non trivial square root of unity modulo N", color="grey")
plt.plot(N[:len(shor_effectiveness)], shor_effectiveness, label="Shor's algorithm - finding period", color="purple")

plt.title("Comparison of effectiveness")
# plt.legend(bbox_to_anchor=(0, 0.92, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.subplots_adjust(bottom=0.35)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
plt.grid(color='gray', linestyle='--', linewidth=0.25)
plt.savefig("./../../../images/plots/effectiveness/effectiveness_ceil_1.png")
