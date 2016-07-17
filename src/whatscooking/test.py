__author__ = 'phx'

def getMap():
    m = {}
    serial = 0
    for i in xrange(0,1000):
        j = i%10
        if j not in m:
            m[j] = serial
            serial+=1
    return m
print(getMap())