import sys
import random

fn_in = sys.argv[1]
fn_out = sys.argv[2]
n_lines = int(sys.argv[3])

data = []
th = 1.
with open(fn_in, encoding='utf-8') as f_in:
	with open(fn_out, "w", encoding='utf-8') as f_out:
		for i, line in enumerate(f_in):
			if random.random() <= th:
				data.append(line)
			if len(data) > n_lines*2:
				print(i, "Capping and reducing sampling rate to", th)
				data = random.sample(data, n_lines)
				th /= 2
		print("Read", i, "lines.")
		data = random.sample(data, n_lines)
		print("Writing", len(data), "lines.")
		for line in data:
			print(line, file=f_out)

