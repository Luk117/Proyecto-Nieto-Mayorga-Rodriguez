import numpy as np
from collections import defaultdict, Counter
import itertools

class Recommender:
    def __init__(self):
        self.RULES = []
        self.frequent_itemsets = None
        self.database = []

    def eclat(self, transactions, minsup_count):
        item_tidsets = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                item_tidsets[item].add(tid)

        item_tidsets = {item: tids for item, tids in item_tidsets.items() if len(tids) >= minsup_count}

        def eclat_recursive(prefix, items_tidsets, frequent_itemsets):
            sorted_items = sorted(items_tidsets.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (item, tidset_i) in enumerate(sorted_items):
                new_itemset = prefix + (item,)
                frequent_itemsets.append((new_itemset, len(tidset_i)))
                suffix_tidsets = {}
                for item_j, tidset_j in sorted_items[i + 1:]:
                    new_tidset = tidset_i & tidset_j
                    if len(new_tidset) >= minsup_count:
                        suffix_tidsets[item_j] = new_tidset
                eclat_recursive(new_itemset, suffix_tidsets, frequent_itemsets)

        frequent_itemsets = []
        eclat_recursive(tuple(), item_tidsets, frequent_itemsets)
        self.frequent_itemsets = frequent_itemsets
        return frequent_itemsets

    def createAssociationRules(self, F, minconf):
        B = []
        for itemset, support in F:
            if len(itemset) > 1:
                for item in itemset:
                    antecedent = set([item])
                    consequent = set(itemset) - antecedent
                    antecedent_support = next((sup for iset, sup in F if set(iset) == antecedent), 0)
                    if antecedent_support > 0:
                        conf = support / antecedent_support
                        if conf >= minconf:
                            sup_b = self.sup(consequent)
                            sup_X = self.sup(antecedent, consequent)
                            r_sup_a = antecedent_support / len(self.database)
                            r_sup_b = sup_b
                            r_sup_X = sup_X

                            lift = conf / r_sup_b if r_sup_b > 0 else 0
                            jaccard = r_sup_X / (r_sup_a + r_sup_b - r_sup_X)
                            conviction = (1 - r_sup_b) / (1 - conf) if (1 - conf) != 0 else float('inf')
                            leverage = r_sup_X - (r_sup_a * r_sup_b)
                            leveraged_lift = leverage / lift if lift != 0 else 0

                            metrics = {
                                'support': support,
                                'confidence': conf,
                                'lift': lift,
                                'leverage': leverage,
                                'jaccard': jaccard,
                                'conviction': conviction,
                                'leveraged_lift': leveraged_lift
                            }
                            B.append((antecedent, consequent, metrics))
        return B

    def sup(self, X, Y=None):
        if Y is None:
            return sum(1 for transaction in self.database if X.issubset(transaction)) / len(self.database)
        else:
            return sum(1 for transaction in self.database if X.issubset(transaction) and Y.issubset(transaction)) / len(self.database)

    def train(self, prices, database) -> None:
        self.database = database
        minsup_count = int(0.05 * len(database)) 
        self.eclat(database, minsup_count)
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf=0.7)
        return self

    def get_recommendations(self, cart, max_recommendations=5):
        recommendations = {}
        for rule in self.RULES:
            if rule[0].issubset(cart):
                for item in rule[1]:
                    if item not in cart:
                        recommendations[item] = recommendations.get(item, 0) + rule[2]['support']
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_recommendations[:max_recommendations]]