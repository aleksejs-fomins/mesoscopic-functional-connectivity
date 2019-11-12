

def sq(x):
    return x**2

data = range(20)

myMap = lambda f, x: list(map(f, x))

print(myMap(sq, data))

