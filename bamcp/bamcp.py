from __future__ import print_function

import random
from . import utils


class BAMCP(object):
    """
    The central BAMCP class, which performs the tree search. It gets a
    root sampler, a tree policy, a default (rollout) policy, a backup strategy,
    decay rate (gamma), epsilon, and maximum reward (R_max).
    """
    def __init__(self, root_sampler, tree_policy, default_policy, backup, gamma, epsilon, R_max):
        self.root_sampler = root_sampler
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.backup = backup
        self.gamma = gamma
        self.epsilon = epsilon
        self.R_max = R_max

    def __call__(self, root, n=1500):
        """
        Run the bayesian adaptive monte carlo tree search.

        :param root: The StateNode
        :param n: The number of samples to be performed
        :return:
        """
        if root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(n):
            self.root_sampler_state = _root_sample(root, self.root_sampler)
            node = _get_next_node(root, self.tree_policy)
            node.reward += self.default_policy(node)
            self.backup(node)

        return utils.rand_max(root.children.values(), key=lambda x: x.q).action


def _expand(state_node):
    action = random.choice(state_node.untried_actions)
    return state_node.children[action].sample_state()


def _best_child(state_node, tree_policy):
    best_action_node = utils.rand_max(state_node.children.values(),
                                      key=tree_policy)
    return best_action_node.sample_state()


def _get_next_node(state_node, tree_policy):
    while not state_node.state.is_terminal():
        if state_node.untried_actions:
            return _expand(state_node)
        else:
            state_node = _best_child(state_node, tree_policy)
    return state_node
