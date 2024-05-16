class Recommender:
    def __init__(self):
        self.RULES = [] 

    def eclat(self, transactions, minsup):
        def eclat_rec(P, minsup, F):
            for V, (Xa, tXa) in enumerate(P):
                F.append((Xa, len(tXa)))
                Pa = []
                for Xb, tXb in P[V+1:]:
                    I = tXa & tXb
                    if len(I) >= minsup:
                        Pa.append((sorted(list(set(Xa + Xb))), I))
                if Pa:
                    eclat_rec(Pa, minsup, F)
            return F

        F = []
        item_sets = [(frozenset([item]), set([idx for idx, transaction in enumerate(transactions) if item in transaction])) for item in set(item for sublist in transactions for item in sublist)]
        return eclat_rec(item_sets, minsup, F)

    def createAssociationRules(self, F, minconf, max_set_size=3):
        def powerset(s):
            result = [[]]
            for elem in s:
                result += [x + [elem] for x in result]
            return result[1:-1]  # exclude empty set and full set

        B = []
        for Z, supZ in [fEntry for fEntry in F if 1 < len(fEntry[0]) <= max_set_size]:
            A = sorted(powerset(list(Z)), key=lambda l: len(l), reverse=True)
            while A:
                X = A.pop(0)
                O = [xEntry for xEntry in F if xEntry[0] == frozenset(X)][0][1]
                c = supZ / O
                if c >= minconf:
                    B.append((X, list(Z - set(X)), supZ, c))
        return B

    """
        This is the class to make recommendations.
        The class must not require any mandatory arguments for initialization.
    """
    def train(self, prices, database, minsup=0.5, minconf=0.7) -> None:

        frequent_itemsets = self.eclat(database, minsup)
        self.RULES = self.createAssociationRules(frequent_itemsets, minconf)

        """
            allows the recommender to learn which items exist, which prices they have, and which items have been purchased together in the past
            :param prices: a list of prices in USD for the items (the item ids are from 0 to the length of this list - 1)
            :param database: a list of lists of item ids that have been purchased together. Every entry corresponds to one transaction
            :return: the object should return itself here (this is actually important!)
        """
        return self

    def get_recommendations(self, cart:list, max_recommendations:int) -> list:
        """
            makes a recommendation to a specific user
            
            :param cart: a list with the items in the cart
            :param max_recommendations: maximum number of items that may be recommended
            :return: list of at most `max_recommendations` items to be recommended
        """
        recommendations = []
        for rule in self.RULES:
            if set(rule[0]).issubset(set(cart)):
             recommendations.extend(rule[1])
            if len(recommendations) >= max_recommendations:
                break
            return recommendations[:max_recommendations]
    
        return [42]  # always recommends the same item (requires that there are at least 43 items)
