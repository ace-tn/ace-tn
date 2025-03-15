from ipeps import Ipeps

if __name__=='__main__':
    dims = {}
    dims['phys'] = 2
    dims['bond'] = 2
    dims['chi'] = 20

    ctmrg_steps = 40

    dtype = "float64"
    device = "cpu"

    ipeps_config = {
        'dtype': dtype,
        'device': device,
        'TN':{
            'dims': dims,
            'nx': 2,
            'ny': 2,
        },
        'model':{
            'name': 'ising',
            'params':{
                'jz': 1.0,
                'hx': 2.95,
            },
        },
        'ctmrg':{
            'steps': ctmrg_steps,
            'projectors': 'half-system',
        },
    }

    ipeps = Ipeps(ipeps_config)

    dtau = 0.1
    steps = 10
    ipeps.evolve(dtau, steps=steps)
    ipeps.measure()

    dtau = 0.01
    steps = 100
    for _ in range(2):
        ipeps.evolve(dtau, steps=steps)
        ipeps.measure()

    dtau = 0.005
    steps = 50
    ipeps.evolve(dtau, steps=steps)
    ipeps.measure()

    dtau = 0.001
    steps = 50
    ipeps.evolve(dtau, steps=steps)
    ipeps.measure()

