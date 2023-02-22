import importlib

class scoring:
    def __init__(self, vars):
        self.__dict__ = vars

    def run(self):
        method_score = importlib.import_module(f'..{self.method}', __name__)
        scoring = method_score.score(self.__dict__)
        self.score = scoring.score
        
        update = scoring.reference()

        self.__dict__ |= update

        return self.__dict__