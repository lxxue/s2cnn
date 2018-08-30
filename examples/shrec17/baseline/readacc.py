import numpy as np

accs = []
for i in range(10):
    with open("log_few10/{}/log.txt".format(i)) as f:
        for line in f:
            if 'Test Acc' in line:
                acc = float(line.split()[-1])
    accs.append(acc)

accs = np.array(accs)
print(accs)
print(np.std(accs))
print(np.mean(accs))
