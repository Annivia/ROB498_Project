import re

def extract_reward(directory):
    pattern = r"(square_bar|square|euclidean_bar|euclidean)"
    match = re.search(pattern, directory)
    if match:
        return match.group(1)
    return None

def extract_algorithm(directory):
    pattern = r"(DDPG|PPO|SAC)"
    match = re.search(pattern, directory)
    if match:
        return match.group(1)
    return None

def extract_checkpoint_id(directory):
    match = re.search(r'checkpoint_0*([1-9]|1[0-5])$', directory)
    if match:
        return int(match.group(1))
    else:
        return None