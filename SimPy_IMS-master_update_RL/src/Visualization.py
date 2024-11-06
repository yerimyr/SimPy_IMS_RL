import matplotlib.pyplot as plt
from config_SimPy import *


def visualization(export_Daily_Report):
    Visual_Dict = {
        'MATERIAL 1': [],
        'WIP': [],
        'Product': [],
        'Keys': {'MATERIAL 1': [], 'WIP': [], 'Product': []}
    }
    Key = ['MATERIAL 1', 'WIP', 'Product']

    for id in I.keys():
        temp = []
        for x in range(SIM_TIME):
            temp.append(export_Daily_Report[x][id*7+6])#Record Onhand inventory at day end
        Visual_Dict[export_Daily_Report[0][id*7+2]].append(temp)#Update 
        Visual_Dict['Keys'][export_Daily_Report[0][2+id*7]].append(export_Daily_Report[0][id *7+1])#Update Keys
    visual = VISUALIAZTION.count(1)
    count_type = 0
    cont_len = 1
    for x in VISUALIAZTION:
        cont = 0
        if x == 1:
            plt.subplot(int(f"{visual}1{cont_len}"))
            cont_len += 1
            for lst in Visual_Dict[Key[count_type]]:
                plt.plot(lst, label=Visual_Dict['Keys'][Key[count_type]][cont])
                plt.legend()
                cont += 1
        count_type += 1
    plt.savefig("Graph")
    plt.clf()
    

