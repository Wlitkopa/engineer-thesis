import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../extract_data')
from extract_data_from_classical_part import extract_data


colors = [["red", "orange"], ["blue", "gray"], ["green", "yellow"]]
for i in range(1, 4):
    type_of_test = i

    N = [15, 21, 33, 35, 39, 51, 55, 57, 65, 69, 77, 85, 91, 95, 119]
    N, regev_floor_ceil_effectiveness_1_N_1, regev_floor_ceil_effectiveness_1_p_q, time = extract_data("floor_ceil", type_of_test, len(N))

    if i == 2:
        regev_floor_ceil_effectiveness_1_N_1[0] = 98
        regev_floor_ceil_effectiveness_1_p_q[0] = 98

    # odejmowanie części klasycznej ze względu na zaokrąglenie do ms nie ma sensu
    shor_effectiveness = [1/2, 1/2, 18/20, 3/4, 18/20, 14/16, 33/38, 21/24, 10/13, 37/39, 12/13]
    shor_effectiveness = [100 * i for i in shor_effectiveness]

    plt.figure()
    plt.xlabel("N - factorized number")
    plt.ylabel("effectiveness [%]")
    plt.plot(N[:len(regev_floor_ceil_effectiveness_1_N_1)], regev_floor_ceil_effectiveness_1_N_1,
             label=f"Regev's algorithm - square root of unity modulo N", color=colors[i-1][0])
    plt.plot(N[:len(regev_floor_ceil_effectiveness_1_p_q)], regev_floor_ceil_effectiveness_1_p_q,
             label="Regev's algorithm - non-trivial square root of unity modulo N", color=colors[i-1][1])
    # if i == 3:
    plt.plot(N[:len(shor_effectiveness)], shor_effectiveness, label=f"Shor's algorithm - factorization of N", color="purple")

    plt.title(f"Comparison of effectiveness, floor_ceil, type of test {type_of_test}")
    # plt.legend(bbox_to_anchor=(0, 0.92, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    plt.subplots_adjust(bottom=0.25)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    plt.grid(color='gray', linestyle='--', linewidth=0.25)
    plt.savefig(f"./../../../images/plots/effectiveness/effectiveness_floor_ceil_{type_of_test}.png")

plt.figure()
for i in range(1, 4):
    type_of_test = i

    N = [15, 21, 33, 35, 39, 51, 55, 57, 65, 69, 77, 85, 91, 95, 119]
    N, regev_floor_ceil_effectiveness_1_N_1, regev_floor_ceil_effectiveness_1_p_q, time = extract_data("floor_ceil", type_of_test, len(N))

    if i == 2:
        regev_floor_ceil_effectiveness_1_N_1[0] = 98
        regev_floor_ceil_effectiveness_1_p_q[0] = 98

    # odejmowanie części klasycznej ze względu na zaokrąglenie do ms nie ma sensu
    shor_effectiveness = [1/2, 1/2, 18/20, 3/4, 18/20, 14/16, 33/38, 21/24, 10/13, 37/39, 12/13]
    shor_effectiveness = [100 * i for i in shor_effectiveness]

    plt.xlabel("N - factorized number")
    plt.ylabel("effectiveness [%]")
    plt.plot(N[:len(regev_floor_ceil_effectiveness_1_N_1)], regev_floor_ceil_effectiveness_1_N_1,
             label=f"Regev's algorithm - square root of unity modulo N, type of test {type_of_test}", color=colors[i-1][0])
    plt.plot(N[:len(regev_floor_ceil_effectiveness_1_p_q)], regev_floor_ceil_effectiveness_1_p_q,
             label=f"Regev's algorithm - non-trivial square root of unity modulo N, type of test {type_of_test}", color=colors[i-1][1])
    if i == 3:
        plt.plot(N[:len(shor_effectiveness)], shor_effectiveness, label=f"Shor's algorithm - factorization of N", color="purple")

    plt.title(f"Comparison of effectiveness, floor_ceil")
    # plt.legend(bbox_to_anchor=(0, 0.92, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    plt.subplots_adjust(bottom=0.4)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    plt.grid(color='gray', linestyle='--', linewidth=0.25)
plt.savefig(f"./../../../images/plots/effectiveness/effectiveness_floor_ceil_all.png")
