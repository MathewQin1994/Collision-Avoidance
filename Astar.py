from PriorityQueue import minheap

class Node():
    def __init__(self,pos,father=None):
        self.pos=pos
        self.father=father

    def __eq__(self, other):
        if self.pos==other.pos:
            return True
        else:
            return False



if __name__=="__main__":
    a=Node((1,2))
    b=Node((12,2))
    print(a!=b)
