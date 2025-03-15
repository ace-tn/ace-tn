from acetn.ipeps import Ipeps
import toml
import csv

def generate(config, input_case, energy_file):
    ipeps = Ipeps(config)

    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.measure()

    ipeps.evolve(dtau=0.01, steps=1000)
    measurements = ipeps.measure()
    ipeps.save(input_case)

    with open(energy_file, 'a') as file:
        file.write(input_case + "," + str(measurements['Energy'].item()) + '\n')

def main(input_cases):
    energy_file = "energies.csv"
    with open(energy_file, 'w'):
        pass

    for input_case in input_cases:
        config = toml.load("../input/" + input_case + ".toml")
        generate(config, input_case, energy_file)

    with open(energy_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input_case = row[0]
            energy = float(row[1])
            print(f"Generated energies in {energy_file}:")
            print(f"Case: {input_case}, Energy: {energy}")

if __name__=='__main__':
    input_cases = ["ising_dims_2_20_dtau_001_hx_295",
                   "heisenberg_dims_3_16_dtau_001"]
    main(input_cases)
