from ctsb import error
from ctsb.utils import seeding

# Unit test for seeding.np_random. Checks that the seed argument is valid.
def test_invalid_seeds():
    for seed in [-1, 'test']:
        try:
            seeding.np_random(seed)
        except error.Error:
            pass
        else:
            assert False, 'Invalid seed {} passed validation'.format(seed)

# Unit test for seeding.np_random. Checks that returned seed is the same as the input seed.
def test_valid_seeds():
    for seed in [0, 1]:
        random, seed1 = seeding.np_random(seed)
        assert seed == seed1


if __name__ == '__main__':
    test_invalid_seeds()
    test_valid_seeds()