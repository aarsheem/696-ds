from gym import envs

for test in envs.registry.all():
    print(test)
