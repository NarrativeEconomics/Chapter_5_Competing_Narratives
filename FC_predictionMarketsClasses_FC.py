import math

import numpy as np
import networkx as nx
import pandas as pd


class agent:
    def __init__(self):
        self.normalize_opinion = 0
        self.opinion = None
        self.alpha = None
        self.opinion_input =None
        self.gamma = None
        self.sigma = None


class TradersGroup:
    def __init__(self, name):
        self.name = name
        self.traders = []

    def add_trader(self, trader):
        self.traders.append(trader)

    def remove_trader(self, trader):
        self.traders.remove(trader)

    def get_traders(self):
        return self.traders

    def average_opinion(self):
        total_opinion = sum([trader.opinion for trader in self.traders])
        return total_opinion / len(self.traders) if self.traders else 0


class opinion_diffusion:
    def __init__(self, param):
        # param: 0: total number of agents, 1: market duration, 2: epsilon, 3: beta 5: agents

        self.N = param[0]
        self.T = param[1]

        self.traders = param[2]
        self.Negative_c = param[3]
        self.Positive_c = param[4]

        # self.G = nx.complete_graph(self.N)

        self.ag_ids = []

        self.curr_day = 0
        self.ag_ids_len = 0

    # select a pair of neighbors
    def select_pair(self):
        '''
        the commented code is needed when using a network structure other than a complete graph

        node = np.random.randint(len(self.G.nodes()))
        while len(self.G.edges(node))<1:
            node = np.random.randint(len(self.G.nodes()))
        pair = np.random.randint(len(self.G.edges(node)))
        self.id1 = list(self.G.nodes)[node]
        self.id2 = [n for n in self.G.neighbors(node)][pair]'''
        self.id1, self.id2 = np.random.choice(range(self.N), 2)

    # update the opinion of a selected pair
    def update_opinion(self, Negative_c, Positive_c, ag):

        timestep = 1

        def odd_tanh(x):
            return 0.5 * (np.tanh(x) - np.tanh(-x))

        def update_trader_opinion(trader, group_opinion, other_group_opinion):
            rate = (-trader.opinion) + odd_tanh(trader.alpha * group_opinion) + trader.opinion_input
            trader.opinion = trader.opinion +rate
            trader.normalize_opinion = (np.tanh(trader.opinion) + 1) / 2

        X_n = Negative_c.average_opinion()
        X_p = Positive_c.average_opinion()

        # Update opinion for agent id1 based on their group
        if ag[self.id1] in Negative_c.get_traders():
            update_trader_opinion(ag[self.id1], X_n, X_p)
        elif ag[self.id1] in Positive_c.get_traders():
            update_trader_opinion(ag[self.id1], X_p, X_n)

        # Update opinion for agent id2 based on their group
        if ag[self.id2] in Negative_c.get_traders():
            update_trader_opinion(ag[self.id2], X_n, X_p)
        elif ag[self.id2] in Positive_c.get_traders():
            update_trader_opinion(ag[self.id2], X_p, X_n)

        return ag

    # pick agents to participate in the market
    def pick_agents(self):

        probs = np.random.uniform(0, 1, self.N)
        p = (self.T - self.curr_day + 1) ** (-3.137) + 0.01  # p = (T -t)^-gamma orignialy 2.44

        self.ag_ids = [i for i, j in enumerate(probs) if j < p]

    # update temporal
    def update_op_series(self, i, ag):
        # self.temporal_op = np.vstack((self.temporal_op, [ag[j].opinion for j in range(len(ag))]))


        self.pick_agents()
        self.ag_ids_len = np.append(self.ag_ids_len, len(self.ag_ids))

        return self.ag_ids

    # control function
    def launch(self, ag):
        self.select_pair()

        ag = self.update_opinion(self.Negative_c, self.Positive_c, ag)
        #ag[self.id1].Normalize_opinion = 1
        #ag[self.id2].normalize_opinion = 1
        return ag


class prediction_market:
    def __init__(self, param):
        # create arrays
        self.pt = np.array([0.5])
        self.last_pt = self.pt[-1]
        # constant parameters
        self.beta = param[0]
        self.gamma = param[1]
        self.ED = 0
        self.temp_dem = 0
        self.ag_part = 0

    # update price
    def update_price(self):
        self.last_pt = self.last_pt + np.round(self.ED, 2)
        if self.last_pt <= 0.01: self.last_pt = 0.01
        #if self.last_pt >= 0.99: self.last_pt = 0.99

    def update_demand(self, ag_idx, net, traders):
        D = 0
        ag_part = 0
        for i in ag_idx:
        #for i in range(0,len(traders)):
            D_temp = traders[i].normalize_opinion - self.last_pt
            if D_temp != 0:
                D += traders[i].normalize_opinion - self.last_pt
                ag_part += 1

        self.ag_part = np.append(self.ag_part, ag_part)

        self.ED = self.beta * D + np.random.normal(0, 0.05)


        self.temp_dem = np.append(self.temp_dem, self.ED)

    def update_price_series(self):
        self.pt = np.append(self.pt, self.last_pt)

    # control function
    def launch(self, ag_idx, net, ag):
        self.update_demand(ag_idx, net, ag )
        self.update_price()
        self.update_price_series()
