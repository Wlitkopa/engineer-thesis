
def is_prime(num):
    flag = True
    if num < 2:
        flag = False
    else:
        for i in range(2, num):
            if num % i == 0:
                flag = False
                break

    return flag