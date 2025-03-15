from .ising import IsingModel
from .heisenberg import HeisenbergModel

__all__ = ['IsingModel', 
           'HeisenbergModel',]

models = {'ising': IsingModel,
          'heisenberg': HeisenbergModel,}
