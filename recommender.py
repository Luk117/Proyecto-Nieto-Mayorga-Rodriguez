class Recommender:
    def __init__(self):
        self.RULES = []
        self.frequent_itemsets = None

    def eclat(self, transactions, minsup):
        item_tidsets = {}
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                item_tidsets[item] = item_tidsets.get(item, 0) + 1

        frequent_items = {item for item, count in item_tidsets.items() if count >= minsup * len(transactions)}
        transactions = [{item for item in t if item in frequent_items} for t in transactions]

        item_tidsets = {}
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                if item in item_tidsets:
                    item_tidsets[item].add(tid)
                else:
                    item_tidsets[item] = {tid}

        def eclat_recursive(prefix, items_tidsets, frequent_itemsets, depth=0, max_depth=3):
            if depth > max_depth:
                return
            for item, tidset in items_tidsets.items():
                new_prefix = prefix + (item,)
                frequent_itemsets.append((new_prefix, len(tidset)))
                suffix_tidsets = {other: tids & tidset for other, tids in items_tidsets.items() if other > item}
                eclat_recursive(new_prefix, suffix_tidsets, frequent_itemsets, depth+1, max_depth)

        frequent_itemsets = []
        eclat_recursive(tuple(), item_tidsets, frequent_itemsets)
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
                            B.append((antecedent, consequent, support, conf))
        return B

    def train(self,prices,database) -> None :
        self.frequent_itemsets = self.eclat(database,minsup=0.05)
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf=0.7)
        return self

    def get_recommendations(self, cart, max_recommendations=5):
        recommendations = {}
        for rule in self.RULES:
            if rule[0].issubset(cart):
                for item in rule[1]:
                    if item not in cart:
                        recommendations[item] = recommendations.get(item, 0) + rule[2]
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_recommendations[:max_recommendations]]


