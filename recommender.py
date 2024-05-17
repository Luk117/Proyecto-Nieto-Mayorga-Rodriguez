import numpy as np
from collections import defaultdict

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

    def calculate_supports(self, D, X, Y=None):
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

    def createAssociationRules(self, F, minconf, transactions):
        B = []
        itemset_support = {frozenset(itemset): support for itemset, support in F}
        for itemset, support in F:
            if len(itemset) > 1:
                for i in range(len(itemset)):
                    antecedent = frozenset(itemset[:i] + itemset[i+1:])
                    consequent = frozenset([itemset[i]])
                    antecedent_support = itemset_support.get(antecedent, 0)
                    if antecedent_support > 0:
                        conf = support / antecedent_support
                        if conf >= minconf:
                            sup_X, sup_XY, sup_Y = self.calculate_supports(transactions, list(antecedent), list(consequent))
                            lift_value = sup_XY / (sup_X * sup_Y) if (sup_X * sup_Y) != 0 else 0
                            leverage_value = sup_XY - (sup_X * sup_Y)
                            jaccard_value = sup_XY / (sup_X + sup_Y - sup_XY) if (sup_X + sup_Y - sup_XY) != 0 else 0
                            B.append((antecedent, consequent, support, conf, lift_value, leverage_value, jaccard_value))
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
        minconf = 0.7  # Using a default minimum confidence of 70%
        minsup_count = int(minsup * len(database))
        
        self.frequent_itemsets = self.eclat(database, minsup_count)
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf, database)
        
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
        
        return [item for item, _ in sorted_recommendations[:max_recommendations]]

# Example usage:
# recommender = Recommender()
# prices = [1.0, 2.0, 3.0, 4.0, 5.0]  # Example prices
# transactions = [[0, 1], [0, 2, 3], [1, 2, 3, 4], [0, 1, 3], [1, 3, 4]]  # Example transactions
# recommender.train(prices, transactions)
# print(recommender.get_recommendations([0, 1], max_recommendations=5))