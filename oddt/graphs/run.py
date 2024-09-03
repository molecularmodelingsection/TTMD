import importlib
from matplotlib import pyplot as plt
import os

class graphs:
    def __init__(self, vars):
        self.__dict__ = vars

        self.value_list = [-1]

        for i in self.output:
            if i == 'eq1' or i == 'eq2':
                pass
            else:
                for v in self.output[i]['results']:
                    self.value_list.append(v)

        self.smooth_list = self.graphic_smooth(self.value_list, self.smooth)

        self.time_list = [0]

        for i in range(1, self.tot_frames + 1):
            self.time_list.append(round(i * self.cfactor, 2))

        self.xticks = []
        self.temperature_list = [self.T_start]

        for set in self.temperature:
            self.xticks.append(set[0])
            
            for i in range(1, int(set[1]/self.cfactor) + 1):
                self.temperature_list.append(set[0])


        self.avg_list = []

        for i in self.output:
            if i != 'eq1' and i != 'eq2':
                self.avg_list.append(self.output[i]['avg'])



    def draw(self):
        method_graphs = importlib.import_module(f'..{self.method}', __name__)
        graphs = method_graphs.graphs(self.__dict__)

        self.__dict__ |= graphs.__dict__

        if self.df == True:
            df_module = importlib.import_module('..df', __name__)
            df = df_module.graphs(self.__dict__)

            self.__dict__ |= df.__dict__

        elif self.df == False:
            self.df_protein = 'Not calculated'
            self.df_prot_h2o = 'Not calculated'

        

    def graphic_smooth(self, value_list, i):
        self.tot_frames = int(self.check_trj_len.ns_to_frame(self.tot_len))
        smooth = round(self.tot_frames / 1000 * i)

        if smooth != 0:
            smooth_sim = []
            for n,v in enumerate(value_list):
                sum = v
                count = 1
                for x in range (1, smooth+1):
                    minor_smooth = n-x
                    major_smooth = n+x
                    
                    if minor_smooth >= 0:
                        sum += value_list[minor_smooth]
                        count += 1

                    if major_smooth < len(value_list):
                        sum += value_list[major_smooth]
                        count += 1

                mean = round(sum / count, 2)
                smooth_sim.append(mean)
            
            return smooth_sim

        else:
            return value_list
