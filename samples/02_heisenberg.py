from ipeps import Ipeps
import toml

if __name__=='__main__':
    ipeps_config = toml.load("./input/02_heisenberg.toml")
    ipeps = Ipeps(ipeps_config)
    ipeps.measure()

    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.measure()

    for _ in range(4):
        ipeps.evolve(dtau=0.01, steps=100)
        ipeps.measure()
