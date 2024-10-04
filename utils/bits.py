def as_bits(x: int, n: int) -> str:
    if x < 0:
        raise ValueError(f'int ({x}) should be >= 0')
    if x.bit_length() > n:
        raise OverflowError(f'int ({x}) too big to represent with {n} bits')
    return f'{x:0{n}b}'


def as_bits_reversed(x: int, n: int) -> str:
    return as_bits(x, n)[::-1]
