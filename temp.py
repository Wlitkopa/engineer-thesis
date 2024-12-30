
ceil_ceil = '''vector with 1 shots: [22, 13, 11]
        vector with 1 shots: [12, 21, 21]
        vector with 1 shots: [11, 21, 14]
        vector with 1 shots: [21, 11, 13]
        vector with 1 shots: [24, 11, 7]
        vector with 1 shots: [27, 11, 15]
        vector with 1 shots: [21, 12, 25]
        vector with 1 shots: [10, 22, 21]
        vector with 1 shots: [16, 10, 11]
        vector with 1 shots: [9, 21, 21]
        vector with 1 shots: [2, 21, 21]
        vector with 1 shots: [11, 19, 21]
        vector with 1 shots: [1, 14, 13]
        vector with 1 shots: [26, 10, 11]
        vector with 1 shots: [11, 22, 20]
        vector with 1 shots: [21, 11, 12]
        vector with 1 shots: [11, 26, 21]
        vector with 1 shots: [22, 11, 9]
        vector with 1 shots: [21, 2, 14]
        vector with 1 shots: [10, 24, 21]
        vector with 1 shots: [21, 11, 6]
        vector with 1 shots: [21, 3, 11]'''

ceil_floor = '''vector with 1 shots: [23, 21, 22]
        vector with 1 shots: [10, 28, 22]
        vector with 1 shots: [11, 23, 21]
        vector with 1 shots: [12, 23, 21]
        vector with 1 shots: [10, 22, 22]
        vector with 1 shots: [22, 11, 10]
        vector with 1 shots: [29, 21, 21]
        vector with 2 shots: [10, 21, 21]
        vector with 2 shots: [11, 21, 23]
        vector with 2 shots: [20, 11, 11]
        vector with 2 shots: [11, 22, 23]
        vector with 2 shots: [21, 9, 11]
        vector with 2 shots: [11, 22, 22]
        vector with 3 shots: [21, 11, 10]
        vector with 3 shots: [21, 12, 11]
        vector with 4 shots: [11, 22, 21]
        vector with 4 shots: [11, 21, 21]
        vector with 5 shots: [11, 21, 22]
        vector with 5 shots: [22, 11, 11]
        vector with 7 shots: [21, 10, 11]
        vector with 13 shots: [21, 11, 11]
        vector with 43 shots: [0, 0, 0]'''


cc_lines = ceil_ceil.splitlines()
cf_lines = ceil_floor.splitlines()

cc_len = len(cc_lines)
cf_len = len(cf_lines)

diff = cc_len - cf_len

if cc_len > cf_len:
    iterNum = cc_len
else:
    iterNum = cf_len

result = ""

# for i in range(iterNum):
#     vect1 = ""
#     vect2 = ""
#
#
#     if i < cc_len:
#         vect1 = cc_lines[i].split(':')[1]
#     if i < cf_len:
#         vect2 = cf_lines[i].split(':')[1]
#
#     if iterNum == cc_len:
#         first = vect1
#         second = vect2
#     else:
#         first = vect2
#         second = vect1
#
#     print(f"first: {first}\nsecond: {second}\n\n")
#
#     result += "{:40s} {:s}\n".format(first[1:], f"{second[1:]}")

print(result)


from_output_reg = '''vector with 1 shots: [7, 26, 1],
vector with 1 shots: [19, 12, 25],
vector with 1 shots: [2, 8, 25],
vector with 1 shots: [13, 29, 4],
vector with 1 shots: [19, 9, 4],
vector with 1 shots: [13, 18, 16],
vector with 1 shots: [25, 14, 16],
vector with 1 shots: [19, 14, 31],
vector with 1 shots: [13, 20, 16],
vector with 1 shots: [6, 27, 31],
vector with 1 shots: [25, 6, 1],
vector with 1 shots: [6, 25, 31],
vector with 1 shots: [6, 26, 25],
vector with 1 shots: [7, 26, 25],
vector with 1 shots: [26, 8, 16],
vector with 1 shots: [20, 13, 25],
vector with 1 shots: [25, 6, 16],
vector with 1 shots: [25, 7, 1],
vector with 1 shots: [26, 6, 25],
vector with 1 shots: [14, 19, 31],
vector with 1 shots: [21, 13, 1],
vector with 1 shots: [20, 13, 16],
vector with 1 shots: [24, 6, 16],
vector with 1 shots: [28, 6, 31],
vector with 1 shots: [6, 25, 25],
vector with 1 shots: [7, 26, 31],
vector with 1 shots: [5, 6, 1],
vector with 1 shots: [13, 24, 1],
vector with 1 shots: [19, 13, 4],
vector with 1 shots: [8, 25, 4],
vector with 1 shots: [3, 25, 16],
vector with 1 shots: [12, 19, 4],
vector with 1 shots: [26, 7, 16],
vector with 1 shots: [19, 10, 16],
vector with 1 shots: [16, 24, 4],
vector with 1 shots: [6, 29, 1],
vector with 1 shots: [25, 1, 1],
vector with 1 shots: [28, 7, 16],
vector with 1 shots: [22, 6, 31],
vector with 2 shots: [6, 26, 16],
vector with 2 shots: [8, 25, 25],
vector with 2 shots: [26, 6, 16],
vector with 2 shots: [26, 7, 1],
vector with 2 shots: [6, 26, 31],
vector with 2 shots: [19, 13, 31],
vector with 2 shots: [6, 24, 25],
vector with 2 shots: [26, 6, 31],
vector with 3 shots: [0, 0, 31],
vector with 3 shots: [13, 19, 31],
vector with 3 shots: [6, 25, 1],
vector with 3 shots: [26, 7, 25],
vector with 3 shots: [19, 13, 25],
vector with 3 shots: [6, 25, 4],
vector with 3 shots: [26, 6, 1],
vector with 4 shots: [0, 0, 4],
vector with 4 shots: [0, 0, 1],
vector with 4 shots: [13, 19, 4],
vector with 4 shots: [13, 19, 1],
vector with 5 shots: [0, 0, 25],
vector with 5 shots: [19, 13, 1],
vector with 5 shots: [19, 13, 16],
vector with 6 shots: [13, 19, 16],
vector with 6 shots: [13, 19, 25],
vector with 9 shots: [0, 0, 16],'''


out_lines = from_output_reg.splitlines()
res = ""

for line in out_lines:
    parts = line.split(',')
    res += f"{parts[0]},{parts[1]}]\n"

print(res)






