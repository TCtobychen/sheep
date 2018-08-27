# params: Initial deposit C, number of sheep N, load L, Allowance A. 
# the number is already the highest allowed. 
# Every healthy sheep consults grass G, people resource for born B, and worth P.
# Every sheep can bring income I by having baby sheep. 

# G, B, P = [,,]
# G + B < I !


def grass_no(C, N, L):
	if C < N * (G + B) + L:
		k = (N * (G + B) - C + L) / (0.8 * P + G + B)
		k = int(k)
		C += k * 0.8 * P
		N -= k
		# After the run 
		return L + N * I
	else:
		return C - N * (G + B) + N * I

def grass_yes(C, N, L, A, x):
	C += A[0]
	if C >= N * (G + B) + L:
		t = C - N * (G + B) + N * L
		t += (1 - w(x)) * A[1]
		t += (1 - w(x)) * (1 - b(x)) * A[2]
		return t
	else:
		ans,p = 0,0
		m = C - N * (G+B) - L
		N = N / (1+x)
		for _x in range(0,x,10):
			t = C + N*(x-_x)*0.8*P+(1-w(_x))*(A[1]+(1-b(_x))*A[2]) - N*(1+_x)*(G+B)-L
			if t > m:
				m=t
				ans = _x
		C += N*(x-_x)*0.8*P- N*(1+_x)*G
		N = N * (1+_x)
		p += w(_x) * winter_no(C,N,L)
		p += (1-w(_x)) * winter_yes(C, N, L, A, _x)


def get_one_year(params, x):
	C, N, L, A = params
	C -= N * x * P
	N += N * x
	ans = 0
	
	# probability is g(x) for being found before buying grass. 
	ans += g(x) * grass_no(C, N, L)
	ans += (1 - g(x)) * grass_yes(C, N, L, A, x)





if __name__ == '__main__':
	params = initial()
	res = []
	for x in range(0,1,100):
		res.append(get_one_year(params, x))
	print(res)