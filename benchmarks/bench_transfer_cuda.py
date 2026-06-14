import time


class MockTensor:
    def __init__(self, shape):
        self.shape = shape

    def to(self, device):
        time.sleep(0.00005)  # mock 50us dispatch overhead
        return self

    @classmethod
    def stack(cls, tensors):
        return cls((len(tensors),) + tensors[0].shape)

    def unbind(self, dim):
        return [MockTensor(self.shape[1:]) for _ in range(self.shape[0])]


def transfer_list(data_list, device):
    return [t.to(device) for t in data_list]


def transfer_batch(data_list, device):
    if not data_list:
        return []
    try:
        stacked = MockTensor.stack(data_list).to(device)
        return stacked.unbind(0)
    except Exception:
        return [t.to(device) for t in data_list]


def run_benchmark():
    data_list = [MockTensor((100, 100)) for _ in range(500)]
    t0 = time.time()
    for _ in range(10):
        transfer_list(data_list, "cuda")
    t_list = time.time() - t0

    t0 = time.time()
    for _ in range(10):
        transfer_batch(data_list, "cuda")
    t_batch = time.time() - t0

    print(f"List Transfer: {t_list:.4f}s")
    print(f"Batch Transfer: {t_batch:.4f}s")
    print(f"Speedup: {t_list / t_batch:.2f}x")


if __name__ == "__main__":
    run_benchmark()
