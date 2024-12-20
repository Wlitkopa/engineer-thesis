import os


def extract_data(parameter, test_type, range_Ns):
    Ns = []
    effectiveness_all = []
    effectiveness_nontrivial = []
    time_in_ms = []
    for filename in os.listdir(f'./../../../output_data/regev/classical_part/type_{test_type}/{parameter}'):
        Ns.append(int(filename.split('_')[-1]))
        with open(f'./../../../output_data/regev/classical_part/type_{test_type}/{parameter}/{filename}') as file:
            content = file.readlines()
            effectiveness_all.append(float(content[10].split(' ')[-1][:-2]))
            effectiveness_nontrivial.append(float(content[11].split(' ')[-1][:-2]))
            time_in_ms.append(int(content[14].split(' ')[-2].split('.')[0]))

    sort_indexes = sorted(range(len(Ns)), key=lambda i: Ns[i])
    Ns = [Ns[i] for i in sort_indexes]
    effectiveness_all = [effectiveness_all[i] for i in sort_indexes]
    effectiveness_nontrivial = [effectiveness_nontrivial[i] for i in sort_indexes]
    time_in_ms = [time_in_ms[i] for i in sort_indexes]

    return Ns[:range_Ns], effectiveness_all[:range_Ns], effectiveness_nontrivial[:range_Ns], time_in_ms[:range_Ns]

if __name__ == "__main__":
    a, b, c, d = extract_data("ceil_ceil", 1, None)
    print(a, b, c, d, sep='\n')