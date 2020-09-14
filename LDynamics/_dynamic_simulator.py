import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import os

T = 1e4
initial_states = [
    [0.45,0.45,0.1],
    [0.4,0.4,0.2],
    [0.35,0.35,0.3],
    [0.3,0.3,0.4],
    [0.25,0.25,0.5],
    [0.2,0.2,0.6],
    [0.15,0.15,0.7],
    [0.10,0.10,0.8],
    [0.05,0.05,0.9],
    
    [0.45,0.1,0.45],
    [0.4,0.2,0.4],
    [0.35,0.3,0.35],
    [0.3,0.4,0.3],
    [0.25,0.5,0.25],
    [0.2,0.6,0.2],
    [0.15,0.7,0.15],
    [0.10,0.8,0.1],
    [0.05,0.9,0.05],
    
    [0.1,0.45,0.45],
    [0.2,0.4,0.4],
    [0.3,0.35,0.35],
    [0.4,0.3,0.3],
    [0.5,0.25,0.25],
    [0.6,0.2,0.2],
    [0.7,0.15,0.15],
    [0.8,0.10,0.1],
    [0.9,0.05,0.05]
]

pre_configured_games = {
    'RPS': [
        [0,-1,1],
        [1,0,-1],
        [-1,1,0]
    ],
    'DHM': [
        [0,-1,-0.5],
        [1,-2,-0.5],
        [0.5,-1.5,-1]
    ],
}

class Population:
    '''
    Defines and evaluates a population follow a given evolution dynamic
    and starting in a initial state
    '''
    def __init__(self,state,payoff_matrix,ed='replicator',strategies_labels=['A','B','C'],eta=0.1):
        '''
        Parameters
        ----------
        state : {list,np.array}
            percentage of population following each strategy in the beginning of simulation
                => must satisfy sum(state) = 1
                => must have an entry for each evaluated strategy
        payoff_matrix : {list,np.ndarray,str}
            game payoff matrix or name of pre-configured game
                => size must be NxN where N is len(state)
                => "RPS","HDM" are pre-configured
        ed : {str}
            evolutionary dynamic to be evaluated
        strategies_labels : {list}
            list of strategy names/labels
        '''
        if abs(sum(state)-1) > 1e-8:
            print('States dont sum 1:',state)
            exit()

        if type(payoff_matrix) is str:
            if payoff_matrix in pre_configured_games.keys():
                payoff_matrix = pre_configured_games[payoff_matrix]

        self.strategies = strategies_labels
        self.state      = copy.copy(state)
        self.payoffs    = copy.copy(payoff_matrix)
        self.F          = [0 for i in self.strategies]
        self.ed         = ed
        self.eta        = eta

    def update_F(self):
        F1 = sum([p*s for p,s in zip(self.payoffs[0],self.state)])
        F2 = sum([p*s for p,s in zip(self.payoffs[1],self.state)])
        F3 = sum([p*s for p,s in zip(self.payoffs[2],self.state)])
        self.F = [F1,F2,F3]

    def run(self):
        self.update_F()
        Fhat = sum(np.array(self.state)*np.array(self.F))

        for i,_ in enumerate(self.strategies):
            xi = self.state[i]
            Fi = self.F[i]

            if self.ed == 'replicator':
                xdot = xi*(Fi - Fhat)

            elif self.ed == 'logit':
                xdot = ( np.exp(Fi/self.eta) / sum([np.exp(self.F[j]/self.eta) for j in range(len(self.strategies))]) ) - xi

            elif self.ed == 'smith':
                d1 = sum([self.state[j]*(self.F[i]-self.F[j]) for j in range(len(self.strategies)) if self.F[j]<self.F[i]])
                d2 = xi*sum([(self.F[j]-self.F[i]) for j in range(len(self.strategies)) if self.F[j]>self.F[i]])
                xdot = d1 - d2

            elif self.ed == 'qlearn':
                xdot = xi*(Fi - Fhat + sum([self.state[j]*np.log(self.state[j]/self.state[i]) for j in range(len(self.strategies))]))

            elif self.ed == 'pgrad':
                f1 = (xi**2)*Fi
                f2 = xi * sum([(self.state[j]**2)*self.F[j] for j in range(len(self.strategies))])
                xdot = (f1 - f2)

            elif self.ed == 'ppo':
                f1 = xi*(Fi - Fhat)
                A = sum(self.F)
                f2 = xi * A * (xi - sum([xj**2 for xj in self.state]))
                xdot = (f1 - f2)
            
            else:
                print('Dynamic not found:',self.ed)

            self.state[i] = max(1e-8,self.state[i]+xdot/300)
        return copy.copy(self.state)

class Evaluator:
    '''
    Evaluates a group of dynamics in a game from multiple initial states
    and save results
    '''
    def __init__(self,payoff_matrix,initial_states=initial_states,strategies=['A','B','C'],T=int(T),
                dynamics=[
                    'replicator','logit','smith','qlearn','pgrad','ppo'
                ]):
        self.models = dynamics
        self.strategies = strategies
        self.T = T
        self.payoffs = payoff_matrix
        self.initial_states = initial_states
        self.results = []

    def evaluate(self):
        '''
        Simulate different models
        '''
        initial_states = self.initial_states

        for mod in self.models:
            history = []
            for is0,s0 in tqdm(enumerate(initial_states),total=len(initial_states),desc='Running {}'.format(mod)):
                h = [s0]
                p = Population(s0,self.payoffs,mod,self.strategies)
                for _ in range(self.T):
                    h.append(p.run())
                h = pd.DataFrame(h,columns=p.strategies)
                h['s0'] = is0
                history.append(h)
            self.results.append([mod,pd.concat(history)])

    def save_results(self,data_path='results//data'):
        '''
        Save simulation results in data_path
        '''
        for r in self.results:
            r[1].to_csv(os.path.join(data_path,'{}.csv'.format(r[0])),index=False)
