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

    def createAssociationRules(self, F, minconf):
        def powerset(s):
            result = [[]]
            for elem in s:
                result += [x + [elem] for x in result]
            return result[1:]

        def diff(first, second):
            second = set(second)
            return [item for item in first if item not in second]

        B = []

        for Z, supZ in [fEntry for fEntry in F if len(fEntry[0]) > 1]:
            A = sorted(powerset(Z), key=lambda l: len(l), reverse=True)
            A = [x for x in A if x and x != Z]
            while A:
                X = A.pop(0)
                O = [xEntry for xEntry in F if xEntry[0] == X][0][1]
                c = supZ / O
                if c >= minconf:
                    B.append((X, diff(Z, X), supZ, c))
                else:
                    m = powerset(X)
                    for z in m:
                        if z in A:
                            A.remove(z)
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
