

class obj():
    def __init__(self, val: int) -> None:
        self.a = val



def test(a: obj):
    a.a = 1

def add(a: int, b: int) -> int:
    return a + b

o = obj(7)
print(o.a)
test(o)
print(o.a)
