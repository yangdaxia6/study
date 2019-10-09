from multiprocessing import Pool
import time

def f(args):
    x, y = args
    time.sleep(2)
    return x*y

if __name__ == '__main__':
    a = range(10)
    b = range(10)[::-1]
    c = []
    for i in range(10):
        c.append([a[i], b[i]])

    with Pool(5) as p:
        d = p.map(f, c)
    print(d)