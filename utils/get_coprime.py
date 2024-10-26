# Large Prime Generation for RSA
import random

# Pre generated primes
first_primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
					31, 37, 41, 43, 47, 53, 59, 61, 67,
					71, 73, 79, 83, 89, 97, 101, 103,
					107, 109, 113, 127, 131, 137, 139,
					149, 151, 157, 163, 167, 173, 179,
					181, 191, 193, 197, 199, 211, 223,
					227, 229, 233, 239, 241, 251, 257,
					263, 269, 271, 277, 281, 283, 293,
					307, 311, 313, 317, 331, 337, 347, 349]


def nBitRandom(n):
	return random.randrange(2**(n-1)+1, 2**n - 1)


def getLowLevelPrime(n, N):
	'''Generate a prime candidate divisible
	by first primes'''
	while True:
		# Obtain a random number
		pc = nBitRandom(n)
		# Test divisibility by pre-generated
		# primes
		for divisor in first_primes_list:
			if pc % divisor == 0 and divisor**2 <= pc:
				break
		else:
			return pc



def isMillerRabinPassed(mrc):
	'''Run 20 iterations of Rabin Miller Primality test'''
	maxDivisionsByTwo = 0
	ec = mrc-1
	while ec % 2 == 0:
		ec >>= 1
		maxDivisionsByTwo += 1
	assert(2**maxDivisionsByTwo * ec == mrc-1)

	def trialComposite(round_tester):
		if pow(round_tester, ec, mrc) == 1:
			return False
		for i in range(maxDivisionsByTwo):
			if pow(round_tester, 2**i * ec, mrc) == mrc-1:
				return False
		return True

	# Set number of trials here
	numberOfRabinTrials = 20
	for i in range(numberOfRabinTrials):
		round_tester = random.randrange(2, mrc)
		if trialComposite(round_tester):
			return False
	return True


if __name__ == '__main__':
	n = 1024
	N = 33

	tries = 100
	for i in range(tries):
		prime_candidate = getLowLevelPrime(n, N)
		if not isMillerRabinPassed(prime_candidate):
			continue
		else:
			print(n, "bit prime is: \n", prime_candidate)
			break

	for i in range(len(first_primes_list)):
		pc = first_primes_list[-(i-1)]
		if pc < N and N % pc == 0:
			pass