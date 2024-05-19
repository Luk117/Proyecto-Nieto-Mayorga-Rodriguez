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

    def createAssociationRules(self, F, minconf, transactions):
        B = []
        itemset_support = {frozenset(itemset): support for itemset, support in F}
        for itemset, support in F:
            if len(itemset) > 1:
                for i in range(len(itemset)):
                    antecedent = frozenset([itemset[i]])
                    consequent = frozenset(itemset[:i] + itemset[i+1:])
                    antecedent_support = itemset_support.get(antecedent, 0)
                    if antecedent_support > 0:
                        conf = support / antecedent_support
                        if conf >= minconf:
                            sup_X, sup_XY, sup_Y = self.calculate_supports(transactions, list(antecedent), list(consequent))
                            lift_value = sup_XY / (sup_X * sup_Y) if (sup_X * sup_Y) != 0 else 0
                            leverage_value = sup_XY - (sup_X * sup_Y)
                            jaccard_value = sup_XY / (sup_X + sup_Y - sup_XY) if (sup_X + sup_Y - sup_XY) != 0 else 0
                            B.append((antecedent, consequent, support, conf, lift_value, leverage_value, jaccard_value))
                            print(f"Rule created: {antecedent} -> {consequent} | conf: {conf}, lift: {lift_value}, leverage: {leverage_value}, jaccard: {jaccard_value}")
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