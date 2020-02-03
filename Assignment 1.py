import pandas as pd
from datetime import datetime
import numpy as np
import math
import matplotlib.pyplot as plt

#x = datetime.strptime(Data.values[0][2], '%m/%d/%Y')

b0_dat = []
b1_dat = []
b2_dat = []
b3_dat = []
b4_dat = []
b5_dat = []
b6_dat = []
b7_dat = []
b8_dat = []
b9_dat = []
b10_dat = []
time = [30,29,26,25,24,23,22,19,18,17]
data = pd.read_excel(r'C:\UofT\fourth\APM466\Bond.xlsx', sheet_name = 'Sheet2')

for i in range(10):
    b0_dat.append([data.values[0][-i-1], time[i]])
    b1_dat.append([data.values[1][-i-1], time[i]])
    b2_dat.append([data.values[2][-i-1], time[i]])
    b3_dat.append([data.values[3][-i-1], time[i]])
    b4_dat.append([data.values[4][-i-1], time[i]])
    b5_dat.append([data.values[5][-i-1], time[i]])
    b6_dat.append([data.values[6][-i-1], time[i]])
    b7_dat.append([data.values[7][-i-1], time[i]])
    b8_dat.append([data.values[8][-i-1], time[i]])
    b9_dat.append([data.values[9][-i-1], time[i]])
    b10_dat.append([data.values[10][-i-1], time[i]])

b0 = pd.DataFrame(b0_dat,columns=['Price','Time'])
b1 = pd.DataFrame(b1_dat,columns=['Price','Time'])
b2 = pd.DataFrame(b2_dat,columns=['Price','Time'])
b3 = pd.DataFrame(b3_dat,columns=['Price','Time'])
b4 = pd.DataFrame(b4_dat,columns=['Price','Time'])
b5 = pd.DataFrame(b5_dat,columns=['Price','Time'])
b6 = pd.DataFrame(b6_dat,columns=['Price','Time'])
b7 = pd.DataFrame(b7_dat,columns=['Price','Time'])
b8 = pd.DataFrame(b8_dat,columns=['Price','Time'])
b9 = pd.DataFrame(b9_dat,columns=['Price','Time'])
b10 = pd.DataFrame(b10_dat,columns=['Price','Time'])

def bond0(p,t):
    semi = (100.75 / (p + 1.5 * (31 - t + 121) / 365)) ** (182.5 / (t + 29)) - 1
    y = semi * 2
    return y


def bond1(p,t):
    y = 0.0075
    if p + 0.75*(31-t+121)/365 > 100:
        while p + 0.75*(31-t+121)/365 - 0.375/((1+y/2)**((t+29)/183)) - 100.375/((1+y/2)**(((t+29)/183)+1)) > 0.0001:
            y -= 0.00001
    else:
        while 0.375/((1+y/2)**((t+29)/183)) + 100.375/((1+y/2)**(((t+29)/183)+1)) - 0.75*(31-t+121)/365 - p >= 0.0001:
            y += 0.00001
    return y


def bond2(p,t):
    y = 0.0075
    if 0.75*(31-t+121)/365 + p > 100:
        while 0.75*(31-t+121)/365 + p - (0.375*((1-1/(1+0.5*y)**2)/(0.5*y)) + (100/(1+0.5*y)**2) + 0.375)/((1+0.5*y)**((t+29)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (0.375*((1-1/(1+0.5*y)**2)/(0.5*y)) + (100/(1+0.5*y)**2) + 0.375)/((1+0.5*y)**((t+29)/183)) - 0.75*(31-t+121)/365 - p > 0.0001:
            y += 0.00001
    return y

def bond3(p,t):
    y = 0.0075
    if 0.75*(31-t+121)/365 + p > 100:
        while 0.75*(31-t+121)/365 + p - (0.375*((1-1/(1+0.5*y)**3)/(0.5*y)) + (100/(1+0.5*y)**3) + 0.375)/((1+0.5*y)**((t+29)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (0.375*((1-1/(1+0.5*y)**3)/(0.5*y)) + (100/(1+0.5*y)**3) + 0.375)/((1+0.5*y)**((t+29)/183)) - 0.75*(31-t+121)/365 - p > 0.0001:
            y += 0.00001
    return y

def bond4(p,t):
    y = 0.005
    if 0.5*(31-t+121)/365 + p > 100:
        while 0.5*(31-t+121)/365 + p - (0.25*((1-1/(1+0.5*y)**4)/(0.5*y)) + (100/(1+0.5*y)**4) + 0.25)/((1+0.5*y)**((t+29)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (0.25*((1-1/(1+0.5*y)**4)/(0.5*y)) + (100/(1+0.5*y)**4) + 0.25)/((1+0.5*y)**((t+29)/183)) - 0.5*(31-t+121)/365 - p > 0.0001:
            y += 0.00001
    return y

def bond5(p,t):
    y = 0.0275
    if 2.75*(31-t-1+31)/365 + p > 100:
        while 2.75*(31-t-1+31)/365 + p - (1.375*((1-1/(1+0.5*y)**4)/(0.5*y)) + 100/(1+0.5*y)**4 + 1.375)/((1+0.5*y)**((t+121)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (1.375*((1-1/(1+0.5*y)**4)/(0.5*y)) + (100/(1+0.5*y)**4) + 1.375)/((1+0.5*y)**((t+121)/183)) - 2.75*(31-t-1+31)/365 - p > 0.0001:
            y += 0.00001
    return y

def bond6(p,t):
    y = 0.0175
    if 1.75*(31-t+121)/365 + p > 100:
        while 1.75*(31-t+121)/365 + p - (0.875*((1-1/(1+0.5*y)**6)/(0.5*y)) + 100/(1+0.5*y)**6 + 0.875)/((1+0.5*y)**((t+29)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (0.875*((1-1/(1+0.5*y)**6)/(0.5*y)) + (100/(1+0.5*y)**6) + 0.875)/((1+0.5*y)**((t+29)/183)) - 1.75*(31-t+121)/365 - p > 0.0001:
            y += 0.00001
    return y

def bond7(p,t):
    y = 0.015
    if 1.5*(31-t-1+31)/365 + p > 100:
        while 1.5*(31-t-1+31)/365 + p - (0.75*((1-1/(1+0.5*y)**6)/(0.5*y)) + (100/(1+0.5*y)**6) + 0.75)/((1+0.5*y)**((t+121)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (0.75*((1-1/(1+0.5*y)**6)/(0.5*y)) + (100/(1+0.5*y)**6) + 0.75)/((1+0.5*y)**((t+121)/183)) - 1.5*(31-t-1+31)/365 - p >= 0.0001:
            y += 0.00001
    return y


def bond8(p,t):
    y = 0.0225
    if 2.25*(31-t+121)/365 + p > 100:
        while 2.25*(31-t+121)/365 + p - (1.125*((1-1/(1+0.5*y)**8)/(0.5*y)) + (100/(1+0.5*y)**8) + 1.125)/((1+0.5*y)**((t+29)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (1.125*((1-1/(1+0.5*y)**8)/(0.5*y)) + (100/(1+0.5*y)**8) + 1.125)/((1+0.5*y)**((t+29)/183)) - 2.25*(31-t+121)/365 - p > 0.0001:
            y += 0.00001
    return y

def bond9(p,t):
    y = 0.015
    if 1.5*(31-t+121)/365 + p > 100:
        while 1.5*(31-t+121)/365 + p - (0.75*((1-1/(1+0.5*y)**9)/(0.5*y)) + (100/(1+0.5*y)**9) + 0.75)/((1+0.5*y)**((t+29)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (0.75*((1-1/(1+0.5*y)**9)/(0.5*y)) + (100/(1+0.5*y)**9) + 0.75)/((1+0.5*y)**((t+29)/183)) - 1.5*(31-t+121)/365 - p >= 0.0001:
            y += 0.00001
    return y

def bond10(p,t):
    y = 0.0125
    if 1.25*(31-t+80)/365 + p > 100:
        while 1.25*(31-t+80)/365 + p - (0.625*((1-1/(1+0.5*y)**10)/(0.5*y)) + (100/(1+0.5*y)**10) + 0.625)/((1+0.5*y)**((t+29)/183)) > 0.0001:
            y -= 0.00001
    else:
        while (0.625*((1-1/(1+0.5*y)**10)/(0.5*y)) + (100/(1+0.5*y)**10) + 0.625)/((1+0.5*y)**((t+29)/183)) - 1.25*(31-t+80)/365 - p >= 0.0001:
            y += 0.00001
    return y

ytm = []
int_ytm = []
year_ytm = []

for i in range(len(time)):
    ytm.append([bond0(b0.values[i][0],b0.values[i][1]),bond1(b1.values[i][0],b1.values[i][1]),bond2(b2.values[i][0],b2.values[i][1]),bond3(b3.values[i][0],b3.values[i][1]),bond4(b4.values[i][0],b4.values[i][1]),bond5(b5.values[i][0],b5.values[i][1]),bond6(b6.values[i][0],b6.values[i][1]),bond7(b7.values[i][0],b7.values[i][1]),bond8(b8.values[i][0],b8.values[i][1]),bond9(b9.values[i][0],b9.values[i][1]),bond10(b10.values[i][0],b10.values[i][1])])

for i in range(len(time)):
    y_05 = ytm[i][0] + (ytm[i][1]-ytm[i][0])*(31+30+31+30+31-time[i])/184
    y_1 = ytm[i][1] + (ytm[i][2]-ytm[1][1])*(30+31+30+31+31+31-time[i])/181
    y_15 = ytm[i][2] + (ytm[i][3]-ytm[i][2])*(31+30+31+30+31-time[i])/184
    y_2 = ytm[i][3] + (ytm[i][4]-ytm[i][3])*(30+31+30+31+31-time[i])/181
    y_25 = ytm[i][5] + (ytm[i][6]-ytm[i][5])*(30+31-time[i])/273
    y_3 = ytm[i][5] + (ytm[i][6]-ytm[i][5])*(30+31+31+30+31+30+31+31-time[i])/273
    y_35 = ytm[i][7] + (ytm[i][8]-ytm[i][7])*(30+31-time[i])/274
    y_4 = ytm[i][7] + (ytm[i][8]-ytm[i][7])*(30+31+31+30+31+30+31+31-time[i])/274
    y_45 = ytm[i][8] + (ytm[i][9]-ytm[i][8])*(31+30+31+30+31-time[i])/184
    y_5 = ytm[i][9] + (ytm[i][10]-ytm[i][9])*(30+31+30+31+31-time[i])/181

    int_ytm.append([y_05,y_1,y_15,y_2,y_25,y_3,y_35,y_4,y_45,y_5])
    year_ytm.append([y_1,y_2,y_3,y_4,y_5])

years = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


plt.subplot(3,1,1)
for i in range(10):
    plt.plot([(time[i]+29)/365,(time[i]+213)/365,(time[i]+394)/365,(time[i]+578)/365,(time[i]+759)/365,(time[i]+852)/365,(time[i]+1124)/365,(time[i]+1217)/365,(time[i]+1489)/365,(time[i]+1673)/365,(time[i]+1854)/365],ytm[i], label = 'Jan'+str(31-time[i]))
plt.xlabel('Year')
plt.ylabel('Yield to Maturity')
plt.title('Yield Curve')
plt.legend()
plt.show()

def z0(p,t):
    semi = (100.75/(p+1.5*(31-t+121)/365))**(182.5/(t+29))-1
    y = semi*2
    return y

def z1(z_0,p,t):
    price = p + 0.75*(31-t+121)/365
    pmt1 = 0.375/(1+0.5*z_0)**((t+29)/182.5)
    semi = (100.375/(price-pmt1))**(182.5/(t+213))-1
    y = semi*2
    return y

def z2(z_0,z_1,p,t):
    price = p + 0.75*(31-t+121)/365
    pmt1 = 0.375/(1+0.5*z_0)**((t+29)/182.5)
    pmt2 = 0.375/(1+0.5*z_1)**((t+213)/182.5)
    semi = (100.375/(price-pmt1-pmt2))**(182.5/(t+394))-1
    y = semi*2
    return y

def z3(z_0,z_1,z_2,p,t):
    price = p + 0.75*(31-t+121)/365
    pmt1 = 0.375/(1+0.5*z_0)**((t+29)/182.5)
    pmt2 = 0.375/(1+0.5*z_1)**((t+213)/182.5)
    pmt3 = 0.375/(1+0.5*z_2)**((t+394)/182.5)
    semi = (100.375/(price-pmt1-pmt2-pmt3))**(182.5/(t+578))-1
    y = semi*2
    return y

def z4(z_0,z_1,z_2,z_3,p,t):
    price = p + 0.5*(31-t+121)/365
    pmt1 = 0.25/(1+0.5*z_0)**((t+29)/182.5)
    pmt2 = 0.25/(1+0.5*z_1)**((t+213)/182.5)
    pmt3 = 0.25/(1+0.5*z_2)**((t+394)/182.5)
    pmt4 = 0.25/(1+0.5*z_3)**((t+578)/182.5)
    semi = (100.25/(price-pmt1-pmt2-pmt3-pmt4))**(182.5/(t+759))-1
    y = semi*2
    return y

def x4(z_0,z_1,z_2,z_3,z_4,p,t):
    x_0 = z_0 + (z_1-z_0)*92/184
    x_1 = z_1 + (z_2-z_1)*91/181
    x_2 = z_2 + (z_3-z_2)*92/184
    x_3 = z_3 + (z_4-z_3)*91/181
    price = 2.75*(31-t-1+31)/365 + p
    pmt1 = 1.375/(1+0.5*x_0)**((t+121)/182.5)
    pmt2 = 1.375/(1+0.5*x_1)**((t+304)/182.5)
    pmt3 = 1.375/(1+0.5*x_2)**((t+486)/182.5)
    pmt4 = 1.375/(1+0.5*x_3)**((t+669)/182.5)
    semi = (101.375/(price-pmt1-pmt2-pmt3-pmt4))**(182.5/(t+852))-1
    y = semi*2
    return y

def z6(z_0,z_1,z_2,z_3,z_4,x_4,p,t):
    price = p + 1.75*(31-t+121)/365
    pmt1 = 0.875/(1+0.5*z_0)**((t+29)/182.5)
    pmt2 = 0.875/(1+0.5*z_1)**((t+213)/182.5)
    pmt3 = 0.875/(1+0.5*z_2)**((t+394)/182.5)
    pmt4 = 0.875/(1+0.5*z_3)**((t+578)/182.5)
    pmt5 = 0.875/(1+0.5*z_4)**((t+759)/182.5)

    z_6 = 0

    while 0.875/(1+0.5*(x_4+(z_6-x_4)*92/273))**((t+943)/182.5) + 100.875/(1+0.5*z_6)**((t+1124)/182.5) - (price-pmt1-pmt2-pmt3-pmt4-pmt5) > 0.0001:
        z_6 += 0.00001

    z_5 = x_4+(z_6-x_4)*92/273
    return z_5, z_6

def x6(z_0,z_1,z_2,z_3,z_4,z_5,z_6,x_4,p,t):
    x_0 = z_0 + (z_1 - z_0) * 92 / 184
    x_1 = z_1 + (z_2 - z_1) * 91 / 181
    x_2 = z_2 + (z_3 - z_2) * 92 / 184
    x_3 = z_3 + (z_4 - z_3) * 91 / 181
    x_5 = z_5 + (z_6 - z_5) * 91 / 181
    price = 1.5*(31-t-1+31)/365 + p
    pmt1 = 0.75/(1+0.5*x_0)**((t+121)/182.5)
    pmt2 = 0.75/(1+0.5*x_1)**((t+304)/182.5)
    pmt3 = 0.75/(1+0.5*x_2)**((t+486)/182.5)
    pmt4 = 0.75/(1+0.5*x_3)**((t+669)/182.5)
    pmt5 = 0.75/(1+0.5*x_4)**((t+852)/182.5)
    pmt6 = 0.75/(1+0.5*x_5)**((t+1036)/182.5)
    semi = (100.75/(price-pmt1-pmt2-pmt3-pmt4-pmt5-pmt6))**(182.5/(t+1217))-1
    y = semi*2
    return y

def z8(z_0,z_1,z_2,z_3,z_4,z_5,z_6,x_6,p,t):
    price = p + 2.25*(31-t+121)/365
    pmt1 = 1.125/(1+0.5*z_0)**((t+29)/182.5)
    pmt2 = 1.125/(1+0.5*z_1)**((t+213)/182.5)
    pmt3 = 1.125/(1+0.5*z_2)**((t+394)/182.5)
    pmt4 = 1.125/(1+0.5*z_3)**((t+578)/182.5)
    pmt5 = 1.125/(1+0.5*z_4)**((t+759)/182.5)
    pmt6 = 1.125/(1+0.5*z_5)**((t+943)/182.5)
    pmt7 = 1.125/(1+0.5*z_6)**((t+1124)/182.5)
    z_8 = 0
    while 1.125/(1+0.5*(x_6+(z_8-x_6)*92/273))**((t+1308)/182.5) + 101.125/(1+0.5*z_8)**((t+1489)/182.5) - (price-pmt1-pmt2-pmt3-pmt4-pmt5-pmt6-pmt7) > 0.0001:
        z_8 += 0.00001

    z_7 = x_6 + (z_8-x_6)*92/373

    return z_7, z_8

def z9(z_0,z_1,z_2,z_3,z_4,z_5,z_6,z_7,z_8,p,t):
    price = p + 1.25*(31-t+80)/365
    pmt1 = 0.75/(1+0.5*z_0)**((t+29)/182.5)
    pmt2 = 0.75/(1+0.5*z_1)**((t+213)/182.5)
    pmt3 = 0.75/(1+0.5*z_2)**((t+394)/182.5)
    pmt4 = 0.75/(1+0.5*z_3)**((t+578)/182.5)
    pmt5 = 0.75/(1+0.5*z_4)**((t+759)/182.5)
    pmt6 = 0.75/(1+0.5*z_5)**((t+943)/182.5)
    pmt7 = 0.75/(1+0.5*z_6)**((t+1124)/182.5)
    pmt8 = 0.75/(1+0.5*z_7)**((t+1308)/182.5)
    pmt9 = 0.75/(1+0.5*z_8)**((t+1489)/182.5)
    semi = (100.75/(price-pmt1-pmt2-pmt3-pmt4-pmt5-pmt6-pmt7-pmt8-pmt9))**(182.5/(t+1673))-1
    y = semi*2
    return y

def z10(z_0,z_1,z_2,z_3,z_4,z_5,z_6,z_7,z_8,z_9,p,t):
    price = p + 1.5*(31-t+121)/365
    pmt1 = 0.625/(1+0.5*z_0)**((t+29)/182.5)
    pmt2 = 0.625/(1+0.5*z_1)**((t+213)/182.5)
    pmt3 = 0.625/(1+0.5*z_2)**((t+394)/182.5)
    pmt4 = 0.625/(1+0.5*z_3)**((t+578)/182.5)
    pmt5 = 0.625/(1+0.5*z_4)**((t+759)/182.5)
    pmt6 = 0.625/(1+0.5*z_5)**((t+943)/182.5)
    pmt7 = 0.625/(1+0.5*z_6)**((t+1124)/182.5)
    pmt8 = 0.625/(1+0.5*z_7)**((t+1308)/182.5)
    pmt9 = 0.625/(1+0.5*z_8)**((t+1489)/182.5)
    pmt10 = 0.625/(1+0.5*z_9)**((t+1673)/182.5)
    semi = (100.625/(price-pmt1-pmt2-pmt3-pmt4-pmt5-pmt6-pmt7-pmt8-pmt9-pmt10))**(182.5/(t+1845))-1
    y = semi*2
    return y

spot = []
spot_year = []
int_spot = []
Forward = []
YEARLY_SPOT = []
for i in range(10):
    z_0 = z0(b0.values[i][0],b0.values[i][1])
    z_1 = z1(z_0,b1.values[i][0],b1.values[i][1])
    z_2 = z2(z_0,z_1,b2.values[i][0],b2.values[i][1])
    z_3 = z3(z_0,z_1,z_2,b3.values[i][0],b3.values[i][1])
    z_4 = z4(z_0,z_1,z_2,z_3,b4.values[i][0],b4.values[i][1])
    x_4 = x4(z_0,z_1,z_2,z_3,z_4,b5.values[i][0],b5.values[i][1])
    z_5, z_6 = z6(z_0,z_1,z_2,z_3,z_4,x_4,b6.values[i][0],b6.values[i][1])
    x_6 = x6(z_0,z_1,z_2,z_3,z_4,z_5,z_6,x_4,b7.values[i][0],b7.values[i][1])
    z_7, z_8 = z8(z_0,z_1,z_2,z_3,z_4,z_5,z_6,x_6,b8.values[i][0],b8.values[i][1])
    z_9 = z9(z_0,z_1,z_2,z_3,z_4,z_5,z_6,z_7,z_8,b9.values[i][0],b9.values[i][1])
    z_10 = z10(z_0,z_1,z_2,z_3,z_4,z_5,z_6,z_7,z_8,z_9,b10.values[i][0],b10.values[i][0])
    spot.append([z_0,z_1,z_2,z_3,z_4,x_4,z_6,x_6,z_8,z_9,z_10])
    yearly_spot = [z_1+(z_2-z_1)*((122+31-time[i])/181),z_3+(z_4-z_3)*((122+31-time[i])/181),z_5+(z_6-z_5)*((122+31-time[i])/181),z_7+(z_8-z_7)*((122+31-time[i])/181),z_9+(z_10-z_9)*((122+31-time[i])/181)]
    forward1 = (((1+0.5*yearly_spot[1])**4)/((1+0.5*yearly_spot[0])**2))**0.5-1
    forward1 = forward1*2
    forward2 = (((1+0.5*yearly_spot[2])**6)/((1+0.5*yearly_spot[0])**2))**0.25-1
    forward2 = forward2*2
    forward3 = (((1+0.5*yearly_spot[3])**8)/((1+0.5*yearly_spot[0])**2))**(1/6)-1
    forward3 = forward3 * 2
    forward4 = (((1+0.5*yearly_spot[4])**10)/((1+0.5*yearly_spot[0])**2))**0.125-1
    forward4 = forward4 * 2
    forward = [forward1,forward2,forward3,forward4]
    Forward.append(forward)
    YEARLY_SPOT.append(yearly_spot)


plt.subplot(3,1,2)
for i in range(10):
    years_spot = [(time[i]+29)/365,(time[i]+213)/365,(time[i]+394)/365,(time[i]+578)/365,(time[i]+759)/365,(time[i]+852)/365,(time[i]+1124)/365,(time[i]+1217)/365,(time[i]+1489)/365,(time[i]+1673)/365,(time[i]+1854)/365]
    plt.plot(years_spot, spot[i], label = 'Jan'+str(31-time[i]))
plt.xlabel('Year')
plt.ylabel('Spot Rate')
plt.title('Spot Curve')
plt.legend()
plt.show()

plt.subplot(3,1,3)
for i in range(10):
    plt.plot(['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'],Forward[i], label = 'Jan'+str(31-time[i]))

plt.xlabel('Year')
plt.ylabel('Forward Rate')
plt.title('Forward Curve')
plt.legend()
plt.show()


#year_ytm

ytm_ret = []
for i in range(5):
    temp = []
    for j in range(9):
        temp.append(math.log(year_ytm[j+1][i]/year_ytm[j][i]))
    ytm_ret.append(temp)

forward_ret = []
for i in range(4):
    temp = []
    for j in range(9):
        temp.append(math.log(Forward[j+1][i]/Forward[j][i]))
    forward_ret.append(temp)

covmat_ytm_ret = np.cov(ytm_ret)
covmat_fwd_ret = np.cov(forward_ret)

#Forward

e_val_ytm, e_vec_ytm = np.linalg.eig(covmat_ytm_ret)
e_val_fwd, e_vec_fwd = np.linalg.eig(covmat_fwd_ret)