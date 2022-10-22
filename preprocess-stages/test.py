import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from src.utils import logger

import numpy as np
import argparse
import igraph
import itertools
import os
import random
import pickle
import pandas as pd

args = {
    "min_active_neighbor": 3,
    "min_inf": 40,
    "max_inf": 1718027,
    "min_deg": 18,
    "max_deg": 483,
    "ego_size": 19,
    "negative": 1,
    "restart_prob": 0.2,
    "walk_length": 1000,
    "output": "/root/data/HeterGAT/stages/stages_subg483_inf_40_1718027_deg_18_483_ego_20_neg_1_restart_20",
    # "graph_file": "/root/TR-pptusn/DeepInf-master/dataset/data/raw-digg/digg_friends.csv",
    # "vote_file":  "/root/TR-pptusn/DeepInf-master/dataset/data/raw-digg/digg_votes1.csv",
}
os.makedirs(args["output"], exist_ok=True)

def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

graph = load_pickle("/root/data/HeterGAT/basic/deg_le483_subgraph.p")
diffusion = load_pickle("/root/data/HeterGAT/basic/deg_le483_df.p")
degree = graph.degree()

def random_walk_with_restart(g, start, restart_prob):
    current = random.choice(start)
    stop = False
    while not stop:
        stop = yield current
        current = random.choice(start) if random.random() < restart_prob or g.degree(current)==0 \
                else random.choice(g.neighbors(current))

class Data:
    def __init__(self):
        self.adj_matrices = []
        self.features = []
        self.vertices = []
        self.labels = []
        self.hashtags = []
        self.stages = []

    def create(self, u, p, t, label, user_affected_now):
        active_neighbor, inactive_neighbor = [], []

        for v in graph.neighbors(u):
            if v in user_affected_now:
                active_neighbor.append(v)
            else:
                inactive_neighbor.append(v)
        if len(active_neighbor) < args["min_active_neighbor"]:
            return

        n = args["ego_size"] + 1
        n_active = 0
        ego = []
        if len(active_neighbor) < args["ego_size"]:
            # we should sample some inactive neighbors
            n_active = len(active_neighbor)
            ego = set(active_neighbor)
            for v in itertools.islice(random_walk_with_restart(graph,
                start=active_neighbor + [u,], restart_prob=args["restart_prob"]), args["walk_length"]):
                if v!=u and v not in ego:
                    ego.add(v)
                    if len(ego) == args["ego_size"]:
                        break
            ego = list(ego)
            if len(ego) < args["ego_size"]:
                return
        else:
            n_active = args["ego_size"]
            samples = np.random.choice(active_neighbor,
                    size=args["ego_size"],
                    replace=False)
            ego += samples.tolist()
        ego.append(u)

        order = np.argsort(ego)
        ranks = np.argsort(order)

        subgraph = graph.subgraph(ego, implementation="create_from_scratch")
        adjacency = np.array(subgraph.get_adjacency().data)
        adjacency = adjacency[ranks][:, ranks]
        self.adj_matrices.append(adjacency)

        feature = np.zeros((n,2))
        for idx, v in enumerate(ego[:-1]):
            if v in user_affected_now:
                feature[idx, 0] = 1
        feature[n-1, 1] = 1
        self.features.append(feature)
        self.vertices.append(np.array(ego, dtype=int))
        self.labels.append(label)
        self.hashtags.append(p)
        self.stages.append(t)

        # circle = subgraph.subgraph(ranks[:n_active], implementation="create_from_scratch")

        if len(self.labels) % 10000 == 0:
            logger.info("Collected %d instances", len(self.labels))

    def dump_data(self):
        self.adj_matrices = np.array(self.adj_matrices)
        self.features = np.array(self.features)
        self.vertices = np.array(self.vertices)
        self.labels = np.array(self.labels)
        self.hashtags = np.array(self.hashtags)
        self.stages = np.array(self.stages)

        output_dir = args["output"]
        with open(os.path.join(output_dir, "adjacency_matrix.npy"), "wb") as f:
            np.save(f, self.adj_matrices)
        with open(os.path.join(output_dir, "influence_feature.npy"), "wb") as f:
            np.save(f, self.features)
        with open(os.path.join(output_dir, "vertex_id.npy"), "wb") as f:
            np.save(f, self.vertices)
        with open(os.path.join(output_dir, "label.npy"), "wb") as f:
            np.save(f, self.labels)
        with open(os.path.join(output_dir, "hashtag.npy"), "wb") as f:
            np.save(f, self.hashtags)
        with open(os.path.join(output_dir, "stage.npy"), "wb") as f:
            np.save(f, self.stages)

        logger.info("Dump %d instances in total" % (len(self.labels)))

        self.adj_matrices = []
        self.features = []
        self.vertices = []
        self.labels = []
        self.hashtags = []
        self.stages = []


    def dump(self):
        logger.info("Dump data ...")

        nu = 0
        for cascade_idx, cascade in diffusion.items():
            nu += 1
            if nu % 100 == 0:
                logger.info("%d (%.2f percent) diffusion processed" % (nu, 100.*nu/len(diffusion)))

            if len(cascade)<args["min_inf"] or len(cascade)>=args["max_inf"]:
                continue
            user_affected_all = set([item[0] for item in cascade])
            user_affected_now = set()
            last = 0
            #infected = set((cas[0][0],))
            for item in cascade[1:]:
                u, t = item
                while last < len(cascade) and cascade[last][1] < t:
                    user_affected_now.add(cascade[last][0])
                    last += 1
                if len(user_affected_now) == 0:
                    continue
                if u in user_affected_now:
                    continue
                stage = (t-cascade[0][1])*8 // (cascade[-1][1]-cascade[0][1])
                if stage < 0:
                    stage = 0
                if stage > 7:
                    stage = 7
                if degree[u]>=args["min_deg"] and degree[u]<args["max_deg"]:
                    # create positive case for user u, photo p, time t
                    self.create(u, cascade_idx, stage, 1, user_affected_now)

                negative = list(set(graph.neighbors(u)) - user_affected_all)
                negative = [v for v in negative \
                        if degree[v]>=args["min_deg"] \
                        and degree[v]<args["max_deg"]]
                if len(negative) == 0:
                    continue
                negative_sample = np.random.choice(negative,
                        size=min(args["negative"], len(negative)), replace=False)
                for v in negative_sample:
                    # create negative case for user v photo p, time t
                    self.create(v, cascade_idx, stage, 0, user_affected_now)
        if len(self.labels) > 0:
            self.dump_data()

data = Data()
data.dump()