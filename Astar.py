from PriorityQueue import minheap

class Node():
    def __init__(self,s,cost,father=None):
        self.s=s
        self.cost=cost
        self.father=father

    def __eq__(self, other):
        if self.cost==other.cost:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.cost<other.cost:
            return True
        else:
            return False
    def __gt__(self, other):
        if self.cost>other.cost:
            return True
        else:
            return False

    def __le__(self, other):
        if self.cost<=other.cost:
            return True
        else:
            return False
    def __ge__(self, other):
        if self.cost>=other.cost:
            return True
        else:
            return False


class OpenList(minheap):
    def __init__(self):
        self._st=set()
        super().__init__()

    def insert(self,item):
        self._heap.append(item)
        self._st.add(item.s)
        self.N+=1
        self.swim(self.N)

    def pop(self):
        it=self._heap[1]
        self._st.remove(it.s)
        self._heap[1]=self._heap.pop()
        self.N-=1
        self.sink(1)
        return it

    def __contains__(self, item):
        if item.s in self._st:
            return True
        else:
            return False


if __name__=="__main__":
    a=Node((1,2),5)
    b=Node((12,2),2)
    c=Node((1,3),3)
    op=OpenList()
    op.insert(a)
    op.insert(b)
    op.insert(c)
