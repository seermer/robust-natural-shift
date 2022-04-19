from torch import nn

def pad_ignore_stride(n):
    if n % 2 != 1:
        raise ValueError('pad_ignore_stride only supports odd kernel size')
    return (n - 1) // 2

def activation(name: str):
    name = name.lower()
    act = {'relu': nn.ReLU(inplace=True), 'elu': nn.ELU(inplace=True),
           'selu': nn.SELU(inplace=True), 'hardswish': nn.Hardswish(inplace=True)}
    if name not in act.keys():
        raise NotImplementedError(f'activation {name} not supported')
    return act[name]


class ArgumentChecker:
    def __init__(self, argv: list):
        self.argv = argv

    def get_requirement(self, *requirements: str):
        for req in requirements:
            req = req.strip().split()
            retval = self._get_req(req)
            if not retval:
                return retval
        return True

    def _get_req(self, requirements):
        try:
            index = self.argv.index(requirements[0])
        except ValueError:
            return False
        if len(requirements) == 1:
            return True
        for i in range(1, len(requirements)):
            if requirements[i] != self.argv[index + i]:
                return False
        return True


class MetricAccumulator:
    def __init__(self, divide_by_step, *metrics):
        self.accumulator = {}
        self.count = {}
        self.divide_by_step = divide_by_step
        self.metrics = metrics
        assert len(self.divide_by_step) == len(self.metrics), 'divide_by_step must have same len as number of metrics'
        self.reset()


    def reset(self):
        for key in self.metrics:
            self.accumulator[key] = 0.
            self.count[key] = 0

    def info(self, steps=0, keys=None):
        if keys is None:
            keys = self.metrics
        retval = ''
        for i, key in enumerate(keys):
            if self.divide_by_step[i] and steps != 0:
                val = round(self.accumulator[key] / steps, 3)
            else:
                val = round(self.accumulator[key], 3)
            retval += '{}:{} '.format(key, val)
        return retval

    def add(self, metric, value):
        self.accumulator[metric] += value
        self.count[metric] += 1

    def get_step(self, key):
        return self.count[key]

