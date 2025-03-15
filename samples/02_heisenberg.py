from ipeps import Ipeps
import toml

if __name__=='__main__':
    ipeps_config = toml.load("./input/02_heisenberg.toml")
    ipeps = Ipeps(ipeps_config)
    ipeps.measure()


    dtau = 0.1
    steps = 10
    ipeps.evolve(dtau=dtau, steps=steps)
    ipeps.measure()

    for _ in range(4):
        dtau = 0.01
        steps = 100
        ipeps.evolve(dtau=dtau, steps=steps)
        ipeps.measure()
