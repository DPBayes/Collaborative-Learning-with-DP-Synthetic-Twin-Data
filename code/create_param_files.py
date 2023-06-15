from itertools import product

# create the parameter file used to invoke infer_models.py in slurm_infer_models.sh
seeds = range(123, 123 + 10)
center_sizes = ["", "0.1", "0.2", "0.5"]
def get_epsilons(center_size):
    if center_size == "0.1": return [1., 2.]
    return [1.]

with open("infer_models_params.txt", "w") as f:
    for (seed, center_size) in product(seeds, center_sizes):
        for eps in get_epsilons(center_size):
            f.writelines(f"{seed} {eps} {center_size}\n")


# create the parameter files used by the dispatching scripts

# Fig 1-3, normal centers
with open("all_centers.txt", "r") as f:
    centers = [c.strip() for c in f.readlines()]
filename = "lls_run_params.txt"

with open(filename, "w") as f:
    for (center, center_size) in product(centers, center_sizes):
        for eps in get_epsilons(center_size):
            f.writelines(f"{center} {eps} {center_size}\n")

# Fig 5, skewed centers
epsilons = [1.]
centers = [("A" + str(fraction)) for fraction in [0., .1, .25, .5, .75, 1.]]
filename = "lls_skewed_run_params.txt"

with open(filename, "w") as f:
    for (center, eps) in product(centers, epsilons):
        f.writelines(f"{center} {eps} \n")
