from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import beta
import numpy as np
import multiprocessing as mtp


# def agent(__a):
#     p = 1-abs(__a-0.5)
#     s = bernoulli.rvs(p)
#     return s
#
#
# if __name__ == "__main__":
#     list_a = np.arange(0.1, 1.1, 0.1)
#     for a in list_a:
#         ss = 0
#         for i in range(1000):
#             ss = ss + agent(a)
#         print(ss/1000)

AvaPrices = np.arange(0.1, 1.1, 0.1)
NPrices = np.zeros(AvaPrices.size)
AvgAward = np.zeros(AvaPrices.size)
DataList = []
NWorker = 0
time = 1
Utility = 0
TotalWorker = 0
ExecTime = 0


def worker(come_time, price):
    global ExecTime
    p = 1 - abs(price-0.5)
    reward = bernoulli.rvs(p)
    exec_time = ExecTime
    return price, reward, come_time + exec_time


# data is a tuple (price, reward, end_time)
def add_data(data):
    index = len(DataList)
    for record in reversed(DataList):
        if record[2] < data[2]:
            index -= 1
        else:
            break
    DataList.insert(index, data)


def data_update(curr_time):
    global NWorker
    global Utility
    if len(DataList) != 0:
        while DataList[-1][2] <= curr_time:
            data = DataList.pop()
            index = (np.where(AvaPrices==data[0]))[0][0]
            AvgAward[index] = (AvgAward[index]*NPrices[index] + data[1])/(NPrices[index]+1)
            NPrices[index] += 1
            NWorker += 1
            Utility += data[1]
            if len(DataList) == 0:
                break


# decide the number of coming workers using the poisson distribution
def come_worker_num():
    mu = 1
    num = poisson.rvs(mu)
    return num


def ucb():
    ucb_index = np.zeros(AvaPrices.size)
    if NWorker > 0:
        for i in range(AvaPrices.size):
            ucb_index[i] = AvgAward[i] + np.sqrt(2*np.log(NWorker)/(NPrices[i]+1e-10))
    index = np.argmax(ucb_index)
    return AvaPrices[index]

def TS():
    TS_index = np.zeros(AvaPrices.size)
    if NWorker > 0:
        for i in range(AvaPrices.size):
            TS_index[i] = beta.rvs(AvgAward[i]*NPrices[i]+1, NPrices[i]-AvgAward[i]*NPrices[i]+1)
    index = np.argmax(TS_index)
    return AvaPrices[index]


def testFramework(T, exec_time):
    global time, TotalWorker, ExecTime
    global NPrices, AvgAward, DataList, NWorker, Utility
    #np.random.seed()
    NPrices = np.zeros(AvaPrices.size)
    AvgAward = np.zeros(AvaPrices.size)
    DataList = []
    NWorker = 0
    time = 1
    Utility = 0
    TotalWorker = 0
    ExecTime = exec_time
    np.random.seed()
    while time <= T:
        # print(time)
        data_update(time)
        n = come_worker_num()
        TotalWorker += n
        for i in range(n):
            #price = ucb()
            price = TS()
            worker_data = worker(time, price)
            add_data(worker_data)
        time += 1
    print("Utility = ", Utility)
    print("Number of Worker = ", TotalWorker)
    return Utility, TotalWorker


def MultiTest(T, exec_time, util_q):
    res = []
    for i in range(10):
        res.append(testFramework(T, exec_time))
    util_q.put(res)


def MultiThreadTest(T, exec_time):
    procs = []
    util_q = mtp.Queue()
    ThreadNum = 2
    for i in range(ThreadNum):
        p = mtp.Process(target=MultiTest, args=(T,exec_time,util_q))
        procs.append(p)
        p.start()

    resultlist = []
    for i in range(ThreadNum):
        resultlist += util_q.get()

    for p in procs:
        p.join()

    return resultlist


def mean_ratio(result_list):
    ratio = 0
    for res in result_list:
        ratio += res[0]/res[1]
    return ratio/len(result_list)

import sys
import pickle
if __name__ == "__main__":
    #exec_delay = 10#int(sys.argv[1])
    data = []
    for exec_delay in np.arange(0,550,50):
        result_list = MultiThreadTest(5000, exec_delay)
        ratio = mean_ratio(result_list)
        data.append((exec_delay, ratio))
    with open('result.data', 'wb') as fp:
        pickle.dump(data, fp)
