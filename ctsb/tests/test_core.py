from ctsb import core

class TestProblem(core.Problem):
    calls = 0

    def __init__(self, arg):
        self.calls += 1
        self.arg = arg

def test_problem():
    env = TestProblem('Hello, World!')
    assert env.arg == 'Hello, World!'
    assert env.calls == 1