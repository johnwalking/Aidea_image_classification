import os
"""
target = os.listdir('./bareland')
for name in os.listdir("./inundated"):
    if name in target:
        print(name)
print("check up done")
"""
for name in os.listdir('./inundated'):
    os.replace('./inundated/'+name, './bareland/'+name)
