# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 22:26:56 2019

@author: nicol
"""

import json
import urllib.request
import csv
import time

vino = 1794
i=1

req = urllib.request.Request("https://www.vivino.com/api/wines/" + str(vino)+ "/reviews?year=null&page=" + str(i)) #CONTROLLA SINTASSI
req.add_header("Accept", "application/json")
jason = urllib.request.urlopen(req).read()
oggettone = json.loads(jason,encoding='UTF-8')
data = oggettone.get("reviews")

keys = data[0].keys()
with open('data1.csv', 'w') as f:
    w=csv.DictWriter(f,keys)
    w.writeheader()
    w.writerows(data)

while i<5000:
    i=i+1
    stringa = "https://www.vivino.com/api/wines/" + str(vino)+ "/reviews?year=null&page=" + str(i)
    req = urllib.request.Request(stringa) #CONTROLLA SINTASSI
    req.add_header("Accept", "application/json")
    jason = urllib.request.urlopen(req).read()
    time.sleep(1)
    try:
        oggettone = json.loads(jason,encoding='UTF-8')
        data = oggettone.get("reviews")
        with open('data1.csv', 'a') as f:
            w=csv.DictWriter(f,keys)
            w.writerows(data)
    except:
        print ("FINITE LE REVIEW, OPPURE ERRORE, A PAGINA " + str(i-1)) #SCRITTO SBAGLIATO, MA STICAZZI


#for item in oggettone["reviews"]:
#   for item in reviews:


#listarevs = oggettone["reviews"]

#for i in range (0, len(listarevs):
#    writer=csv.writer("dati.csv", dialect ='excel')
#    writer.writerow (listarevs[i])
    

#for i in range (0, len(oggettone))