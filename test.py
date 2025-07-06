import json
import os
import settings
from network.utils import create_network_from_file
from data import x_train, y_train

if __name__ == "__main__":
    cache_file_path = os.path.join(settings.BASE_DIR, "data", "exported_network.json")

    with open(cache_file_path) as file:
        raw_cache_data = json.load(file)

    network = create_network_from_file(raw_cache_data)

    x_train, y_train = network.preprocess(x_train, y_train)

    for x, y in zip(x_train, y_train):
        scores, number = network.compute(x)
        assert number == y, f"Number: {number} is not right for y: {y}"
