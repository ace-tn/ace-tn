from acetn.ipeps import Ipeps
import toml

def main(config):
    # initialize an iPEPS
    ipeps = Ipeps(config)

    # Evolve for a few steps at large dtau=0.1
    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.measure()

    # Evolve for many more steps at smaller dtau=0.01
    for _ in range(5):
        ipeps.evolve(dtau=0.01, steps=100)
        ipeps.measure()

if __name__=='__main__':
    # Set config options in the input file: "02_heisenberg.toml"
    config = toml.load("./input/02_heisenberg.toml")
    main(config)
