class Recommender:
    def __init__(self):
        self.RULES = []
        self.frequent_itemsets = None

    def eclat(self, transactions, minsup, max_itemset_size=3):
        def eclat_rec(P, minsup, F, depth=1):
            if depth > max_itemset_size:
                return
            for V, (Xa, tXa) in enumerate(P):
                if len(tXa) >= minsup:
                    F.append((Xa, len(tXa)))
                    Pa = []
                    for Xb, tXb in P[V+1:]:
                        I = tXa & tXb
                        if len(I) >= minsup:
                            new_set = Xa | Xb
                            if len(new_set) <= max_itemset_size:
                                Pa.append((new_set, I))
                    eclat_rec(Pa, minsup, F, depth + 1)

        item_sets = [(frozenset([item]), set(idx for idx, transaction in enumerate(transactions) if item in transaction))
                     for item in set().union(*transactions)]
        F = []
        eclat_rec(item_sets, minsup, F)
        return F

    def createAssociationRules(self, F, minconf):
        B = []
        for itemset, support in F:
            if len(itemset) > 1:
                for item in itemset:
                    antecedent = frozenset([item])
                    consequent = itemset - antecedent
                    antecedent_support = next((sup for iset, sup in F if iset == antecedent), 0)
                    if antecedent_support > 0:
                        conf = support / antecedent_support
                        if conf >= minconf:
                            B.append((antecedent, consequent, support, conf))
        return B

    def train(self, database, minsup=0.05, minconf=0.7, max_itemset_size=3):
        self.frequent_itemsets = self.eclat(database, minsup, max_itemset_size)
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf)
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
