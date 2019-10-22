# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:24:51 2019

@author: berkc
"""


import pandas as pd
import matplotlib.pyplot as plt
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

# =============================================================================
# RANDOM SELECTION
# =============================================================================
#import random

#N = 10000
#d = 10
#toplam = 0
#secilenler = []
#for n in range(0,N):
#    ad = random.randrange(d)
#    secilenler.append(ad)
#    odul = veriler.values[n,ad]#n.satır = 1 odul = 1
#    toplam = toplam + odul
#
#
#plt.hist(secilenler)
#plt.show()


# =============================================================================
# UCB
# =============================================================================

N = 10000 # 10.000 tıklama/reklam gösterimi
d = 10 # toplam 10 ilan
#Ri(n)
oduller = [0] * d # ilk başta bütün ilanları ödülü 0
#Ni(n)
tiklamalar = [0] * d # o ana kadar ki tıklamalar
toplam2 = 0 # toplam ödül
secilenler2 = []
for n in range(0,N):
    ad = 0 # seçilen ilan
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * (math.log(n)/tiklamalar[i]))
            ucb = ortalama + delta
        else :
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    secilenler2.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n,ad]
    oduller[ad] = oduller[ad] + odul
    toplam2 = toplam2 + odul
    
print('Toplam odul : ',toplam2 )

plt.hist(secilenler2)
plt.show()


