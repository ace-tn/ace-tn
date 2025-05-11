from acetn.ipeps import Ipeps

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
    # Basic config for the TFIM with hx=2.95
    config = {
        'dtype': "float64",
        'device': "cpu",
        'TN':{
            'nx': 2,
            'ny': 2,
            'dims': {'phys':2, 'bond':2, 'chi':20},
        },
        'model':{
            'name': 'ising',
            'params':{
                'jz': 1.0,
                'hx': 2.95,
            },
        },
    }

    main(config)
