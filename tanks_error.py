from spear import *
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy
from statistics import mean


a = 0.5
az0 = 0.75
az12 = 0.75
az23 = 0.75
g = 9.81
t_step = 0.1 #duration of a tick in seconds
q_max = 6.0
q_step = q_max/5.0
q_med = q_max/2.0
l_max = 20.0
l_min = 0.0

l_goal = 10.0
delta_l = 0.5
epsilon = 0.3


def compute_flow_rate(x1, x2, a, a12):
    if x1 > x2:
        return a12*a*numpy.sqrt(2*g)*numpy.sqrt(x1-x2)
    else:
        return -a12*a*numpy.sqrt(2*g)*numpy.sqrt(x2-x1)


def compute_q12(ds):
    l1 = ds['l1']
    l2 = ds['l2']
    return compute_flow_rate(l1, l2, a, az12)


def compute_q23(ds):
    l2 = ds['l2']
    l3 = ds['l3']
    return compute_flow_rate(l2, l3, a, az23)


def compute_q0(ds):
    return az0*a*numpy.sqrt(2*g)*numpy.sqrt(ds['l3'])


def step(fun_q1, fun_q2, fun_q3, ds):
    newds = {}
    q1 = ds['q1']
    q2 = ds['q2']
    q3 = ds['q3']
    q12 = compute_q12(ds)
    q23 = compute_q23(ds)
    #q0 = compute_q0(ds)
    newds['l1'] = max(0.0 , ds['l1']+q1*t_step-q12*t_step)
    newds['l2'] = max(0.0 , ds['l2']+q12*t_step-q23*t_step)
    newds['l3'] = max(0.0 , ds['l3']+q2*t_step+q23*t_step-q3*t_step)
    newds['q1'] = fun_q1(ds)
    newds['q2'] = fun_q2(ds)
    newds['q3'] = fun_q3(ds)
    return newds


def run(fun_q1, fun_q2, fun_q3, ds, k):
    res = []
    ds2 = ds
    for i in range(k):
        res.append(ds2)
        ds2 = step(fun_q1, fun_q2, fun_q3, ds2)
    return res


def simulate(fun_q1, fun_q2, fun_q3, ds, n, l, k):
    data = [ [] for i in range(k) ]
    for i in range(n*l):
        sample = run(fun_q1, fun_q2, fun_q3, ds, k)
        for j in range(k):
            data[j].append(sample[j])
    return data

def q1_scenario_1(q, delta):
    return lambda ds: max(0.0, rnd.normal(q, delta))


def q1_scenario_2(q, delta):
    return lambda ds: min(max(0.0, ds['q2']+rnd.normal(q, delta)), q_max)


def controller_q1(ds, l, q, delta):
    if ds['l1'] > l+delta:
        return max(0.0, ds['q1']-q)
    if ds['l1'] < l-delta:
        return min(q_max, ds['q1']+q)
    return ds['q1']


def controller_q3(ds, l, q, delta):
    if ds['l3'] > l+delta:
        return min(q_max, ds['q3'] + q)
    elif ds['l3'] < l-delta:
        return max(0.0, ds['q3'] - q)
    else:
        return ds['q3']


def init_ds(q1, q2, q3, l1, l2, l3):
    return {'l1': l1, 'l2': l2, 'l3': l3, 'q1': q1, 'q2': q2, 'q3': q3}


q2_start = 3.0
q2_dev = 0.5

ds_start = init_ds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
k = 150
l = 10


err100 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 10, l, k)
err500 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 50, l, k)
err1000 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 100, l, k)
err5000 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 500, l, k)
err10000 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)

lstl3_100 = [ [] for i in range(k) ]
standev_100 = [ [] for i in range(k) ]
standerr_100 = [ [] for i in range(k) ]
zscorel3_100 = [ [] for i in range(k) ]

lstl3_500 = [ [] for i in range(k) ]
standev_500 = [ [] for i in range(k) ]
standerr_500 = [ [] for i in range(k) ]
zscorel3_500 = [ [] for i in range(k) ]

lstl1_1000 = [ [] for i in range(k) ]
lstl2_1000 = [ [] for i in range(k) ]
lstl3_1000 = [ [] for i in range(k) ]
standev_1000 = [ [] for i in range(k) ]
standerr_1000 = [ [] for i in range(k) ]
zscorel1_1000 = [ [] for i in range(k) ]
zscorel2_1000 = [ [] for i in range(k) ]
zscorel3_1000 = [ [] for i in range(k) ]

lstl3_5000 = [ [] for i in range(k) ]
standev_5000 = [ [] for i in range(k) ]
standerr_5000 = [ [] for i in range(k) ]
zscorel3_5000 = [ [] for i in range(k) ]

lstl3_10000 = [ [] for i in range(k) ]
standev_10000 = [ [] for i in range(k) ]
standerr_10000 = [ [] for i in range(k) ]
zscorel3_10000 = [ [] for i in range(k) ]


for i in range(k):
    lstl3_100[i] = list(map(lambda ds: ds['l3'], err100[i]))
    standev_100[i] = numpy.std(lstl3_100[i])
    standerr_100[i] = standev_100[i]/ numpy.sqrt(100)
    lstl3_500[i] = list(map(lambda ds: ds['l3'], err500[i]))
    standev_500[i] = numpy.std(lstl3_500[i])
    standerr_500[i] = standev_500[i]/ numpy.sqrt(500)
    lstl3_1000[i] = list(map(lambda ds: ds['l3'], err1000[i]))
    standev_1000[i] = numpy.std(lstl3_1000[i])
    standerr_1000[i] = standev_1000[i]/ numpy.sqrt(1000)
    lstl3_5000[i] = list(map(lambda ds: ds['l3'], err5000[i]))
    standev_5000[i] = numpy.std(lstl3_5000[i])
    standerr_5000[i] = standev_5000[i]/ numpy.sqrt(5000)
    lstl3_10000[i] = list(map(lambda ds: ds['l3'], err10000[i]))
    standev_10000[i] = numpy.std(lstl3_10000[i])
    standerr_10000[i] = standev_10000[i]/ numpy.sqrt(10000)
    
fix, ax = plt.subplots()
ax.plot(range(0,k),standev_100,label="N = 100")
ax.plot(range(0,k),standev_500,label="N = 500")
ax.plot(range(0,k),standev_1000,label="N = 1000")
ax.plot(range(0,k),standev_5000,label="N = 5000")
ax.plot(range(0,k),standev_10000,label="N = 10000")
legend = ax.legend()
plt.title("Standard deviation")
plt.savefig("SD.png")
plt.show()

fix, ax = plt.subplots()
ax.plot(range(0,k),standerr_100,label="N = 100")
ax.plot(range(0,k),standerr_500,label="N = 500")
ax.plot(range(0,k),standerr_1000,label="N = 1000")
ax.plot(range(0,k),standerr_5000,label="N = 5000")
ax.plot(range(0,k),standerr_10000,label="N = 10000")
legend = ax.legend()
plt.title("Standard error of the mean")
plt.savefig("SEM.png")
plt.show()


print("I will now proceed to compute several simulations to obtain the analysis of the error, please wait")

expected1 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected2 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected3 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected4 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected5 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected6 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected7 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected8 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected9 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected10 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected11 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected12 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected13 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected14 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected15 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected16 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected17 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected18 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected19 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)
expected20 = simulate(lambda ds: controller_q1(ds, l_goal, q_step, delta_l), q1_scenario_2(0,1), lambda ds: controller_q3(ds, l_goal, q_step, delta_l), ds_start, 1000, l, k)


print("Simulations completed. Next we compute the expected value of the (real) distribution")

expectedl1_1 = [ [] for i in range(k) ]
expectedl2_1 = [ [] for i in range(k) ]
expectedl3_1 = [ [] for i in range(k) ]

expectedl1_2 = [ [] for i in range(k) ]
expectedl2_2 = [ [] for i in range(k) ]
expectedl3_2 = [ [] for i in range(k) ]

expectedl1_3 = [ [] for i in range(k) ]
expectedl2_3 = [ [] for i in range(k) ]
expectedl3_3 = [ [] for i in range(k) ]

expectedl1_4 = [ [] for i in range(k) ]
expectedl2_4 = [ [] for i in range(k) ]
expectedl3_4 = [ [] for i in range(k) ]

expectedl1_5 = [ [] for i in range(k) ]
expectedl2_5 = [ [] for i in range(k) ]
expectedl3_5 = [ [] for i in range(k) ]

expectedl1_6 = [ [] for i in range(k) ]
expectedl2_6 = [ [] for i in range(k) ]
expectedl3_6 = [ [] for i in range(k) ]

expectedl1_7 = [ [] for i in range(k) ]
expectedl2_7 = [ [] for i in range(k) ]
expectedl3_7 = [ [] for i in range(k) ]

expectedl1_8 = [ [] for i in range(k) ]
expectedl2_8 = [ [] for i in range(k) ]
expectedl3_8 = [ [] for i in range(k) ]

expectedl1_9 = [ [] for i in range(k) ]
expectedl2_9 = [ [] for i in range(k) ]
expectedl3_9 = [ [] for i in range(k) ]

expectedl1_10 = [ [] for i in range(k) ]
expectedl2_10 = [ [] for i in range(k) ]
expectedl3_10 = [ [] for i in range(k) ]

expectedl1_11 = [ [] for i in range(k) ]
expectedl2_11 = [ [] for i in range(k) ]
expectedl3_11 = [ [] for i in range(k) ]

expectedl1_12 = [ [] for i in range(k) ]
expectedl2_12 = [ [] for i in range(k) ]
expectedl3_12 = [ [] for i in range(k) ]

expectedl1_13 = [ [] for i in range(k) ]
expectedl2_13 = [ [] for i in range(k) ]
expectedl3_13 = [ [] for i in range(k) ]

expectedl1_14 = [ [] for i in range(k) ]
expectedl2_14 = [ [] for i in range(k) ]
expectedl3_14 = [ [] for i in range(k) ]

expectedl1_15 = [ [] for i in range(k) ]
expectedl2_15 = [ [] for i in range(k) ]
expectedl3_15 = [ [] for i in range(k) ]

expectedl1_16 = [ [] for i in range(k) ]
expectedl2_16 = [ [] for i in range(k) ]
expectedl3_16 = [ [] for i in range(k) ]

expectedl1_17 = [ [] for i in range(k) ]
expectedl2_17 = [ [] for i in range(k) ]
expectedl3_17 = [ [] for i in range(k) ]

expectedl1_18 = [ [] for i in range(k) ]
expectedl2_18 = [ [] for i in range(k) ]
expectedl3_18 = [ [] for i in range(k) ]

expectedl1_19 = [ [] for i in range(k) ]
expectedl2_19 = [ [] for i in range(k) ]
expectedl3_19 = [ [] for i in range(k) ]

expectedl1_20 = [ [] for i in range(k) ]
expectedl2_20 = [ [] for i in range(k) ]
expectedl3_20 = [ [] for i in range(k) ]

for i in range (0,k):
    expectedl1_1[i] = list(map(lambda ds: ds['l1'], expected1[i]))
    expectedl1_2[i] = list(map(lambda ds: ds['l1'], expected2[i]))
    expectedl1_3[i] = list(map(lambda ds: ds['l1'], expected3[i]))
    expectedl1_4[i] = list(map(lambda ds: ds['l1'], expected4[i]))
    expectedl1_5[i] = list(map(lambda ds: ds['l1'], expected5[i]))
    expectedl1_6[i] = list(map(lambda ds: ds['l1'], expected6[i]))
    expectedl1_7[i] = list(map(lambda ds: ds['l1'], expected7[i]))
    expectedl1_8[i] = list(map(lambda ds: ds['l1'], expected8[i]))
    expectedl1_9[i] = list(map(lambda ds: ds['l1'], expected9[i]))
    expectedl1_10[i] = list(map(lambda ds: ds['l1'], expected10[i]))
    expectedl1_11[i] = list(map(lambda ds: ds['l1'], expected11[i]))
    expectedl1_12[i] = list(map(lambda ds: ds['l1'], expected12[i]))
    expectedl1_13[i] = list(map(lambda ds: ds['l1'], expected13[i]))
    expectedl1_14[i] = list(map(lambda ds: ds['l1'], expected14[i]))
    expectedl1_15[i] = list(map(lambda ds: ds['l1'], expected15[i]))
    expectedl1_16[i] = list(map(lambda ds: ds['l1'], expected16[i]))
    expectedl1_17[i] = list(map(lambda ds: ds['l1'], expected17[i]))
    expectedl1_18[i] = list(map(lambda ds: ds['l1'], expected18[i]))
    expectedl1_19[i] = list(map(lambda ds: ds['l1'], expected19[i]))
    expectedl1_20[i] = list(map(lambda ds: ds['l1'], expected20[i]))   
    expectedl2_1[i] = list(map(lambda ds: ds['l2'], expected1[i]))
    expectedl2_2[i] = list(map(lambda ds: ds['l2'], expected2[i]))
    expectedl2_3[i] = list(map(lambda ds: ds['l2'], expected3[i]))
    expectedl2_4[i] = list(map(lambda ds: ds['l2'], expected4[i]))
    expectedl2_5[i] = list(map(lambda ds: ds['l2'], expected5[i])) 
    expectedl2_6[i] = list(map(lambda ds: ds['l2'], expected6[i]))
    expectedl2_7[i] = list(map(lambda ds: ds['l2'], expected7[i]))
    expectedl2_8[i] = list(map(lambda ds: ds['l2'], expected8[i]))
    expectedl2_9[i] = list(map(lambda ds: ds['l2'], expected9[i]))
    expectedl2_10[i] = list(map(lambda ds: ds['l2'], expected10[i])) 
    expectedl2_11[i] = list(map(lambda ds: ds['l2'], expected11[i]))
    expectedl2_12[i] = list(map(lambda ds: ds['l2'], expected12[i]))
    expectedl2_13[i] = list(map(lambda ds: ds['l2'], expected13[i]))
    expectedl2_14[i] = list(map(lambda ds: ds['l2'], expected14[i]))
    expectedl2_15[i] = list(map(lambda ds: ds['l2'], expected15[i])) 
    expectedl2_16[i] = list(map(lambda ds: ds['l2'], expected16[i]))
    expectedl2_17[i] = list(map(lambda ds: ds['l2'], expected17[i]))
    expectedl2_18[i] = list(map(lambda ds: ds['l2'], expected18[i]))
    expectedl2_19[i] = list(map(lambda ds: ds['l2'], expected19[i]))
    expectedl2_20[i] = list(map(lambda ds: ds['l2'], expected20[i]))    
    expectedl3_1[i] = list(map(lambda ds: ds['l3'], expected1[i]))
    expectedl3_2[i] = list(map(lambda ds: ds['l3'], expected2[i]))
    expectedl3_3[i] = list(map(lambda ds: ds['l3'], expected3[i]))
    expectedl3_4[i] = list(map(lambda ds: ds['l3'], expected4[i]))
    expectedl3_5[i] = list(map(lambda ds: ds['l3'], expected5[i]))  
    expectedl3_6[i] = list(map(lambda ds: ds['l3'], expected6[i]))
    expectedl3_7[i] = list(map(lambda ds: ds['l3'], expected7[i]))
    expectedl3_8[i] = list(map(lambda ds: ds['l3'], expected8[i]))
    expectedl3_9[i] = list(map(lambda ds: ds['l3'], expected9[i]))
    expectedl3_10[i] = list(map(lambda ds: ds['l3'], expected10[i]))
    expectedl3_11[i] = list(map(lambda ds: ds['l3'], expected11[i]))
    expectedl3_12[i] = list(map(lambda ds: ds['l3'], expected12[i]))
    expectedl3_13[i] = list(map(lambda ds: ds['l3'], expected13[i]))
    expectedl3_14[i] = list(map(lambda ds: ds['l3'], expected14[i]))
    expectedl3_15[i] = list(map(lambda ds: ds['l3'], expected15[i]))  
    expectedl3_16[i] = list(map(lambda ds: ds['l3'], expected16[i]))
    expectedl3_17[i] = list(map(lambda ds: ds['l3'], expected17[i]))
    expectedl3_18[i] = list(map(lambda ds: ds['l3'], expected18[i]))
    expectedl3_19[i] = list(map(lambda ds: ds['l3'], expected19[i]))
    expectedl3_20[i] = list(map(lambda ds: ds['l3'], expected20[i]))
 
expected_mean1 = [ [] for i in range(k) ]
expected_mean2 = [ [] for i in range(k) ]
expected_mean3 = [ [] for i in range(k) ]

for j in range (k):
    expected_mean1[j] = mean([mean(expectedl1_1[j]),mean(expectedl1_2[j]),mean(expectedl1_3[j]),mean(expectedl1_4[j]),mean(expectedl1_5[j]),mean(expectedl1_6[j]),mean(expectedl1_7[j]),mean(expectedl1_8[j]),mean(expectedl1_9[j]),mean(expectedl1_10[j]),mean(expectedl1_11[j]),mean(expectedl1_12[j]),mean(expectedl1_13[j]),mean(expectedl1_14[j]),mean(expectedl1_15[j]),mean(expectedl1_16[j]),mean(expectedl1_17[j]),mean(expectedl1_18[j]),mean(expectedl1_19[j]),mean(expectedl1_20[j])])
    expected_mean2[j] = mean([mean(expectedl2_1[j]),mean(expectedl2_2[j]),mean(expectedl2_3[j]),mean(expectedl2_4[j]),mean(expectedl2_5[j]),mean(expectedl2_6[j]),mean(expectedl2_7[j]),mean(expectedl2_8[j]),mean(expectedl2_9[j]),mean(expectedl2_10[j]),mean(expectedl2_11[j]),mean(expectedl2_12[j]),mean(expectedl2_13[j]),mean(expectedl2_14[j]),mean(expectedl2_15[j]),mean(expectedl2_16[j]),mean(expectedl2_17[j]),mean(expectedl2_18[j]),mean(expectedl2_19[j]),mean(expectedl2_20[j])])
    expected_mean3[j] = mean([mean(expectedl3_1[j]),mean(expectedl3_2[j]),mean(expectedl3_3[j]),mean(expectedl3_4[j]),mean(expectedl3_5[j]),mean(expectedl3_6[j]),mean(expectedl3_7[j]),mean(expectedl3_8[j]),mean(expectedl3_9[j]),mean(expectedl3_10[j]),mean(expectedl3_11[j]),mean(expectedl3_12[j]),mean(expectedl3_13[j]),mean(expectedl3_14[j]),mean(expectedl3_15[j]),mean(expectedl3_16[j]),mean(expectedl3_17[j]),mean(expectedl3_18[j]),mean(expectedl3_19[j]),mean(expectedl3_20[j])])

print("Mean computed. Finally we evaluate the z-scores")

for i in range (0,10):
    zscorel3_100[i] = 0
    zscorel3_500[i] = 0
    zscorel3_1000[i] = 0
    zscorel3_5000[i] = 0
    zscorel3_10000[i] = 0
    
for i in range (10,k):
    zscorel3_100[i] = (mean(lstl3_100[i]) - expected_mean3[i]) / standerr_100[i]
    zscorel3_500[i] = (mean(lstl3_500[i]) - expected_mean3[i]) / standerr_500[i]
    zscorel3_1000[i] = (mean(lstl3_1000[i]) - expected_mean3[i]) / standerr_1000[i]
    zscorel3_5000[i] = (mean(lstl3_5000[i]) - expected_mean3[i]) / standerr_5000[i]
    zscorel3_10000[i] = (mean(lstl3_10000[i]) - expected_mean3[i]) / standerr_10000[i]

limit1 = [ [1.96] for i in range(k) ]
limit2 = [ [-1.96] for i in range(k) ]

fix, ax = plt.subplots()
ax.plot(range(0,k),zscorel3_100,label="N = 100")
ax.plot(range(0,k),zscorel3_1000,label="N = 1000")
ax.plot(range(0,k),zscorel3_10000,label="N = 10000")
ax.plot(range(0,k),limit1, 'r--')
ax.plot(range(0,k),limit2, 'r--')
legend = ax.legend()
plt.xlim([10, k])
plt.title("Value of z-score in time")
plt.savefig("zScore.png")
plt.show()

fix, ax = plt.subplots()
ax.plot(range(0,k),zscorel3_100,label="N = 100")
ax.plot(range(0,k),zscorel3_500,label="N = 500")
ax.plot(range(0,k),zscorel3_1000,label="N = 1000")
ax.plot(range(0,k),zscorel3_5000,label="N = 5000")
ax.plot(range(0,k),zscorel3_10000,label="N = 10000")
ax.plot(range(0,k),limit1, 'r--')
ax.plot(range(0,k),limit2, 'r--')
legend = ax.legend()
plt.xlim([10, k])
plt.title("Value of z-score in time")
plt.savefig("zScore_all5.png")
plt.show()



for i in range(k): 
    lstl1_1000[i] = list(map(lambda ds: ds['l1'], err1000[i]))
    lstl2_1000[i] = list(map(lambda ds: ds['l2'], err1000[i]))   

for i in range (0,10):
    zscorel1_1000[i] = 0
    zscorel2_1000[i] = 0
    
for i in range (10,k):
    zscorel1_1000[i] = (mean(lstl1_1000[i]) - expected_mean1[i])*numpy.sqrt(1000)/numpy.std(lstl1_1000[i])
    zscorel2_1000[i] = (mean(lstl2_1000[i]) - expected_mean2[i])*numpy.sqrt(1000)/numpy.std(lstl2_1000[i])
    
fix, ax = plt.subplots()
ax.plot(range(0,k),zscorel1_1000,label="z-score on l1")
ax.plot(range(0,k),zscorel2_1000,label="z-score on l2")
ax.plot(range(0,k),zscorel3_1000,label="z-score on l3")
legend = ax.legend()
plt.xlim([10, k])
plt.title("Comparison of the z-scores for the levels of water (N = 1000)")
plt.savefig("total.png")
plt.show()
