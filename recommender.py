class Recommender:
    def __init__(self):
        self.RULES = []
        self.frequent_itemsets = None

    def eclat(self, transactions, minsup):
        item_tidsets = {}
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                if item in item_tidsets:
                    item_tidsets[item].add(tid)
                else:
                    item_tidsets[item] = {tid}

        item_tidsets = {item: tids for item, tids in item_tidsets.items() if len(tids) >= minsup}

        def eclat_recursive(prefix, items_tidsets, frequent_itemsets):
            sorted_items = sorted(items_tidsets.items())
            for i, (item, tidset_i) in enumerate(sorted_items):
                new_itemset = prefix + (item,)
                frequent_itemsets.append((new_itemset, len(tidset_i)))
                suffix_tidsets = {}
                for item_j, tidset_j in sorted_items[i + 1:]:
                    new_tidset = tidset_i & tidset_j
                    if len(new_tidset) >= minsup:
                        suffix_tidsets[item_j] = new_tidset
                eclat_recursive(new_itemset, suffix_tidsets, frequent_itemsets)

        frequent_itemsets = []
        eclat_recursive(tuple(), item_tidsets, frequent_itemsets)
        return frequent_itemsets

    def createAssociationRules(self, F, minconf):
        B = []
        for itemset, support in F:
            if len(itemset) > 1:
                for item in itemset:
                    antecedent = tuple([item])
                    consequent = tuple(itemset) - antecedent
                    antecedent_support = next((sup for iset, sup in F if tuple(iset) == antecedent), 0)
                    if antecedent_support > 0:
                        conf = support / antecedent_support
                        if conf >= minconf:
                            B.append((antecedent, consequent, support, conf))
        return B

    def train(self,prices,database) -> None :
        self.frequent_itemsets = self.eclat(database,minsup=0.7)
        self.RULES = self.createAssociationRules(self.frequent_itemsets, minconf=0.5)
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


