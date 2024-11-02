
from implementations.r_haner import HanerRegev as Regev


# Ns = [15, 21, 33, 35, 39, 51, 55, 57, 65, 69, 77, 85, 91, 95, 119, 143]
# d_qd_list = [[True, True], [True, False], [False, True], [False, False]]


shots_num = 128
regev = Regev(shots_num)


Ns = [15]
d_qd_list = [[False, False]]
regev.run_quantum_part(Ns, d_qd_list)


file_name = "/home/koan/myHome/AGH/PracaInżynierska/pycharm_github/shor_mmik/output_data/regev/quantum_part/ceil_ceil/N_51"
regev.run_file_data_analyzer(file_name)


