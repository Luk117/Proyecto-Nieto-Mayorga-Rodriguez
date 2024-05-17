import numpy as np
from collections import defaultdict
import itertools

def calculate_supports(D, X, Y=None):
    count_X = 0
    count_XY = 0
    count_Y = 0 if Y else None
    
    for transaction in D:
        if set(X).issubset(transaction):
            count_X += 1
            if Y and set(Y).issubset(transaction):
                count_XY += 1
        if Y and set(Y).issubset(transaction):
            count_Y += 1
    
    sup_X = count_X / len(D)
    sup_XY = count_XY / len(D)
    sup_Y = count_Y / len(D) if count_Y is not None else None
    
    return sup_X, sup_XY, sup_Y

def conf(D, X, Y, count_X, count_XY):
    return count_XY / count_X if count_X != 0 else 0

def lift(sup_X, sup_Y, sup_XY):
    return sup_XY / (sup_X * sup_Y) if (sup_X * sup_Y) != 0 else 0

def leverage(sup_X, sup_Y, sup_XY):
    return sup_XY - (sup_X * sup_Y)

def jaccard(D, X, Y, count_X, count_XY, count_Y):
    return count_XY / (count_X + count_Y - count_XY) if (count_X + count_Y - count_XY) != 0 else 0

def getRuleMetric(D, X, Y, metric):
    sup_X, sup_XY, sup_Y = calculate_supports(D, X, Y)
    count_X = sup_X * len(D)
    count_XY = sup_XY * len(D)
    count_Y = sup_Y * len(D) if sup_Y is not None else None
    
    metrics = {
        'sup': sup_XY,
        'conf': conf(D, X, Y, count_X, count_XY),
        'lift': lift(sup_X, sup_Y, sup_XY),
        'leverage': leverage(sup_X, sup_Y, sup_XY),
        'jaccard': jaccard(D, X, Y, count_X, count_XY, count_Y),
    }
    return metrics[metric]

class Recommender:
    """
    This is the class to make recommendations.
    The class must not require any mandatory arguments for initialization.
    """
    
    def __init__(self):
        self.RULES = []
        self.frequent_itemsets = None
        self.prices = None

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
        return frequent_itemsets

    def createAssociationRules(self, F, minconf, database):
        B = []
        itemset_support = {frozenset(itemset): support for itemset, support in F}
        for itemset, support in F:
            if len(itemset) > 1:
                for i in range(len(itemset)):
                    antecedent = frozenset([itemset[i]])
                    consequent = frozenset(itemset[:i] + itemset[i+1:])
                    antecedent_support = itemset_support.get(antecedent, 0)
                    if antecedent_support > 0:
                        conf_value = support / antecedent_support
                        if conf_value >= minconf:
                            lift_value = getRuleMetric(database, list(antecedent), list(consequent), 'lift')
                            leverage_value = getRuleMetric(database, list(antecedent), list(consequent), 'leverage')
                            jaccard_value = getRuleMetric(database, list(antecedent), list(consequent), 'jaccard')
                            B.append((antecedent, consequent, support, conf_value, lift_value, leverage_value, jaccard_value))
        return B

    def train(self, prices, database) -> None:
        """
        Allows the recommender to learn which items exist, which prices they have, and which items have been purchased together in the past
        :param prices: a list of prices in USD for the items (the item ids are from 0 to the length of this list - 1)
        :param database: a list of lists of item ids that have been purchased together. Every entry corresponds to one transaction
        :return: the object should return itself here (this is actually important!)
        """
        self.prices = prices
        minsup = 0.05  # Using a default minimum support of 5%
        minconf = 0.2  # Using a default minimum confidence of 70%
        minsup_count = int(minsup * len(database))
        
        print("Calculating frequent itemsets...")
        self.frequent_itemsets = self.eclat(database, minsup_count)
        print("Frequent itemsets:", self.frequent_itemsets)
        
        print("Creating association rules...")
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf, database)
        print("Association rules:", self.RULES)
        
        return self

    def get_recommendations(self, cart: list, max_recommendations: int) -> list:
        """
        Makes a recommendation to a specific user
        :param cart: a list with the items in the cart
        :param max_recommendations: maximum number of items that may be recommended
        :return: list of at most `max_recommendations` items to be recommended
        """
        recommendations = defaultdict(float)
        for rule in self.RULES:
            if rule[0].issubset(cart):
                for item in rule[1]:
                    if item not in cart:
                        # Use a combination of confidence, lift, leverage, and jaccard to prioritize recommendations
                        score = (rule[3] + rule[4] + rule[5] + rule[6]) * self.prices[item]
                        recommendations[item] += score
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Debug print to verify recommendations
        print("Recommendations:", sorted_recommendations)
        
        return [item for item, _ in sorted_recommendations[:max_recommendations]]