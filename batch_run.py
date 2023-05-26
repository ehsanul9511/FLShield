import subprocess
import hashlib
from multiprocessing import Pool

def calculate_hash(string):
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Convert the string to bytes and update the hash object
    sha256_hash.update(string.encode('utf-8'))

    # Get the hexadecimal representation of the hash value
    hash_value = sha256_hash.hexdigest()

    return hash_value

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    # output, error = process.communicate()
    return 



def run_commands(commands, result_dir):
    # generate a hash for each command to use as exp_id
    orig_commands = commands
    exp_ids = [calculate_hash(command) for command in commands]
    commands = [command + " --hash=exp_id " + exp_id for command, exp_id in zip(commands, exp_ids)]

    processes = []
    # run commands
    for command in commands:
        process = subprocess.Popen(command, shell=True)
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()



    # run commands in parallel
    # with Pool() as pool:
    #     pool.map(run_command, commands)

    # join all the commands into one command by &
    # comb = " & ".join(commands)
    # run_command(comb)

    # all the results are in the saved_models/exp_id folder
    # copy the results to the saved_results/result_dir/orig_command[15:].replace(' ', '__') folder

    run_command(f"mkdir -p saved_results/{result_dir}")
    for command, exp_id in zip(orig_commands, exp_ids):
        run_command(f"cp -rf saved_models/{exp_id} saved_results/{result_dir}/{command[15:].replace(' ', '__')}")


def contrib_adjustment_experiment():
    root_command = "python main.py --attack_methods=dba --injective_florida --epochs=201"

    values = [0, 0.5, 1, 2, 5]
    values = [i * 0.1 for i in range(21)]
    # values = [1]
    commands = [root_command + f" --contrib_adjustment={c}" for c in values]
    run_commands(commands, "contrib_adjustment")


if __name__ == '__main__':
    contrib_adjustment_experiment() 