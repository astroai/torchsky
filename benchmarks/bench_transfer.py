import time
import torch


def transfer_list(data_list, device):
    return [t.to(device) for t in data_list]


def transfer_batch(data_list, device):
    if not data_list:
        return []
    try:
        if len(data_list) == 1:
            return [data_list[0].to(device)]
        stacked = torch.stack(data_list)
        transferred = stacked.to(device)
        return list(transferred.unbind(0))
    except Exception:
        return [t.to(device) for t in data_list]


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Simulating the tensor data typically output from read_images_batch
    data_list = [torch.randn(100, 100) for _ in range(500)]

    # Warmup
    for _ in range(5):
        transfer_list(data_list, device)
        transfer_batch(data_list, device)

    t0 = time.perf_counter()
    for _ in range(50):
        transfer_list(data_list, device)
    t_list = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(50):
        transfer_batch(data_list, device)
    t_batch = time.perf_counter() - t0

    print(f"List Transfer: {t_list:.4f}s")
    print(f"Batch Transfer: {t_batch:.4f}s")
    if t_batch < t_list:
        print(f"Speedup: {t_list / t_batch:.2f}x")
    else:
        print(
            f"Slowdown: {t_batch / t_list:.2f}x (expected on CPU without copy overhead)"
        )


if __name__ == "__main__":
    run_benchmark()
