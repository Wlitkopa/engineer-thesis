import matplotlib.pyplot as plt
import sys

N = [15, 21, 33, 35, 39, 51, 55, 57]

normal_t_2_all = [0, 84, 64, 52, 82, 0, 34, 82]
normal_t_2_non_trivial = [0, 36, 54, 22, 80, 0, 22, 70]
random_t_2_all = [100, 98, 68, 70, 98, 100, 36, 50]
random_t_2_non_trivial = [82, 76, 48, 50, 44, 100, 26, 42]

normal_t_3_all = [0, 90, 72, 60, 86, 0, 36, 66]
normal_t_3_non_trivial = [0, 48, 54, 22, 80, 0, 20, 60]
random_t_3_all = [100, 96, 72, 76, 96, 100, 40, 66]
random_t_3_non_trivial = [88, 68, 56, 58, 60, 100, 28, 44]

normal_t_5_all = [100, 88, 62, 80, 100, 0, 46, 54]
normal_t_5_non_trivial = [100, 58, 40, 44, 100, 0, 28, 42]
random_t_5_all = [100, 90, 78, 72, 100, 100, 60, 58]
random_t_5_non_trivial = [90, 64, 62, 46, 48, 40, 20, 32]

normal_t_8_all = [96, 94, 82, 72, 100, 100, 76, 56]
normal_t_8_non_trivial = [96, 70, 42, 28, 24, 98, 14, 26]
random_t_8_all = [100, 96, 76, 68, 100, 100, 52, 60]
random_t_8_non_trivial = [100, 66, 48, 40, 44, 34, 26, 36]

normal_t_14_all = [100, 94, 66, 74, 78, 66, 48, 58]
normal_t_14_non_trivial = [100, 74, 46, 48, 44, 54, 24, 36]
random_t_14_all = [98, 98, 84, 84, 78, 46, 44, 64]
random_t_14_non_trivial = [98, 88, 50, 64, 48, 32, 22, 46]

normal_t_16_all = [72, 84, 80, 62, 58, 66, 52, 68]
normal_t_16_non_trivial = [16, 64, 54, 46, 30, 36, 26, 52]
random_t_16_all = [98, 100, 84, 68, 70, 64, 56, 68]
random_t_16_non_trivial = [96, 94, 48, 44, 34, 40, 34, 48]

# t_list = [2, 5, 8, 14, 16]
t_list = [3]
for t in t_list:
    plt.figure()
    plt.xlabel("N - factorized number")
    plt.ylabel("effectiveness [%]")
    plt.plot(N, globals()[f'normal_t_{t}_all'], label=f"Square root of unity modulo N, type of test 1", color='red')
    plt.plot(N, globals()[f'normal_t_{t}_non_trivial'], label="Non-trivial square root of unity modulo N, type of test 1",
             color='orange')
    plt.plot(N, globals()[f'random_t_{t}_all'], label=f"Square root of unity modulo N, type of test 3", color='green')
    plt.plot(N, globals()[f'random_t_{t}_non_trivial'], label="Non-trivial square root of unity modulo N, type of test 3",
             color='yellow')
    plt.title(f"Effectiveness of Regev's algorithm for t = {t}")
    plt.subplots_adjust(bottom=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    plt.grid(color='gray', linestyle='--', linewidth=0.25)
    plt.savefig(f"./../../../images/plots/effectiveness_t/effectiveness_t_{t}.png")