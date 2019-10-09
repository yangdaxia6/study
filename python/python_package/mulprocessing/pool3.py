import time
from multiprocessing import Pool
def run(fn, gn):
  #fn: 函数参数是数据列表的一个元素
  time.sleep(1)
  print(fn, gn)
  return str(fn)+'+'+str(gn)

def writerun(resf, fns, gns):
  ww = ''
  for fn, gn in zip(fns, gns):
    res = run(fn,gn)
    ww += res + '\n'
  with open(resf)as f:
    f.write(ww)


if __name__ == "__main__":
  testFL = range(20)
  test_two = [testFL, testFL]

  print('shunxu:') #顺序执行(也就是串行执行，单进程)
  s = time.time()
  '''
  for fn in testFL:
    run(fn, fn*5)
  '''


  e1 = time.time()
  print("顺序执行时间：", int(e1 - s))

  print('concurrent:') #创建多个进程，并行执行
  pool = Pool(8)  #创建拥有5个进程数量的进程池
  #testFL:要处理的数据列表，run：处理testFL列表中数据的函数
  for i in testFL:
    a = pool.apply_async(run, args=(i, i*5))
    print(a)
  pool.close()#关闭进程池，不再接受新的进程
  pool.join()#主进程阻塞等待子进程的退出
  e2 = time.time()
  print("并行执行时间：", int(e2-e1))
