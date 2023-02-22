import os
import importlib

class system_preparation:
    def __init__(self, vars):
        self.__dict__ = vars

        if self.iso == 'yes':
            self._iso = 'iso'
        elif vars['iso'] == 'no':
            self._iso = ''


    def prepare(self):
        print('\n——System preparation')
        if not os.path.exists('prep'):
            os.mkdir('prep')

        os.chdir('prep')
        
        method_prep = importlib.import_module(f'..{self.method}', __name__)   
        prep = method_prep.prepare(self.__dict__)
        update = prep.prepare()

        self.__dict__ |= update
        
        os.chdir('..')

    #     statistics()