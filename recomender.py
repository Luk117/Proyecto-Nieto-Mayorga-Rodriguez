import itertools
import time
from collections import defaultdict
from itertools import chain, combinations

class Recommender:
    def __init__(self):
        self.itemsets = defaultdict(int)
        self.rules = []

    def train(self, database):
        start_time = time.time()
        min_support = 0.01
        min_confidence = 0.5

        # Generate frequent itemsets using Eclat algorithm
        def eclat(prefix, items, min_support):
            while items:
                item, transactions = items.pop()
                support = len(transactions)
                if support >= min_support:
                    self.itemsets[prefix + (item,)] = support
                    suffix = []
                    for other_item, other_transactions in items:
                        intersection = transactions & other_transactions
                        if len(intersection) >= min_support:
                            suffix.append((other_item, intersection))
                    eclat(prefix + (item,), suffix, min_support)

        # Create a transaction list
        transactions = list(map(set, database))
        items = sorted((item, {i for i, transaction in enumerate(transactions) if item in transaction})
                       for item in set(chain.from_iterable(transactions)))

        eclat((), items, min_support * len(transactions))

        # Generate association rules
        for itemset in self.itemsets:
            if len(itemset) > 1:
                for antecedent in chain(*[combinations(itemset, r) for r in range(1, len(itemset))]):
                    antecedent = tuple(sorted(antecedent))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    confidence = self.itemsets[itemset] / self.itemsets[antecedent]
                    if confidence >= min_confidence:
                        self.rules.append((antecedent, consequent, confidence))

        print(f"Training completed in {time.time() - start_time} seconds")

    def get_recommendations(self, cart):
        start_time = time.time()
        recommendations = defaultdict(float)
        cart = set(cart)
        for antecedent, consequent, confidence in self.rules:
            if set(antecedent).issubset(cart):
                for item in consequent:
                    if item not in cart:
                        recommendations[item] += confidence

        sorted_recommendations = sorted(recommendations, key=recommendations.get, reverse=True)
        print(f"Recommendation generation took {time.time() - start_time} seconds")
        return sorted_recommendations