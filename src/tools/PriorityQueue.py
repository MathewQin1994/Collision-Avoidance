

class maxheap():
    def __init__(self):
        self._heap=[None]
        self.N=0

    def __len__(self):
        return self.N

    def __iter__(self):
        return iter(self._heap[1:])

    def __str__(self):
        n=1
        result=""
        layer=1
        while n<=self.N:
            if n==2**layer:
                result+="\n"
                layer+=1
            result+=str(self._heap[n])+" "
            n+=1
        return result

    def swim(self,k):
        while k>1 and self._heap[k//2]<self._heap[k]:
            self._heap[k],self._heap[k//2]=self._heap[k//2],self._heap[k]
            k=k//2

    def sink(self,k):
        while 2*k<=self.N:
            j=2*k
            if j<self.N and self._heap[j]<self._heap[j+1]:
                j+=1
            if self._heap[k]>=self._heap[j]:
                break
            self._heap[k], self._heap[j] = self._heap[j], self._heap[k]
            k=j

    def insert(self,item):
        self._heap.append(item)
        self.N+=1
        self.swim(self.N)

    def pop(self):
        it = self._heap[1]
        if self.N==1:
            self._heap.pop()
            self.N -= 1
            return it
        self._heap[1]=self._heap.pop()
        self.N-=1
        self.sink(1)
        return it

    def peek(self):
        return self._heap[1]


class minheap(maxheap):
    def swim(self,k):
        while k>1 and self._heap[k//2]>self._heap[k]:
            self._heap[k],self._heap[k//2]=self._heap[k//2],self._heap[k]
            k=k//2

    def sink(self,k):
        while 2*k<=self.N:
            j=2*k
            if j<self.N and self._heap[j]>self._heap[j+1]:
                j+=1
            if self._heap[k]<=self._heap[j]:
                break
            self._heap[k], self._heap[j] = self._heap[j], self._heap[k]
            k=j





if __name__=="__main__":
    mh=minheap()
    mh.insert(1)
    mh.pop()