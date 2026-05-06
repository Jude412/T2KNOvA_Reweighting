import numpy as np

        arrays = [np.load(f) for f in input]
        combined = np.concatenate(arrays, axis=0)
        np.save(output[0], combined)