from stream_al.selection_strategies.baseline import RandomSelection

Wtheta = 0.1
Hloc = 0.88
class Random():

    def labeling(self,theta):
        qs = RandomSelection()
        if theta >= qs.utility():
            return True
        else:
            return False
    def special_cluster(self,special_factor):
        if special_factor > Wtheta:
            return True
        else:
            return False

    def les(self,ent):
        if ent > Hloc:
            return True
        else:
            return False






