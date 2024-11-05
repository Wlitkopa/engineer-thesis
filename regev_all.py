
from implementations.r_haner import HanerRegev as Regev


# All analysed values
# Ns = [15, 21, 33, 35, 39, 51, 55, 57, 65, 69, 77, 85, 91, 95, 119, 143]
# d_qd_list = [[True, True], [True, False], [False, True], [False, False]]

# Initiating Regev algorithm class
shots_num = 128
regev = Regev(shots_num)

# Running quantum part
Ns = [15]
d_qd_list = [[False, False]]
# regev.run_quantum_part(Ns, d_qd_list)

# Analysing data from a quantum part output file (new version)
# file_name = "/home/koan/myHome/AGH/PracaInżynierska/pycharm_github/shor_mmik/output_data/regev/quantum_part/ceil_ceil/N_51"
Ns = [15, 21, 33, 35, 39, 51, 55, 57]
d_qd_list = [[True, True], [True, False], [False, True], [False, False]]
number_of_combinations = 1000
regev.run_file_data_analyzer_new(Ns, d_qd_list, number_of_combinations)

# Analysing data from a quantum part output file (old version)
file_name = "/home/koan/myHome/AGH/PracaInżynierska/pycharm_github/shor_mmik/output_data/regev/quantum_part/ceil_ceil/N_51"
# regev.run_file_data_analyzer_old(file_name)

# if you want to run Regev on quantum computer
N = 15
d_ceil_bool = True
qd_ceil_bool = True
# regev.run_on_quantum_computer(N, d_ceil_bool, qd_ceil_bool)


