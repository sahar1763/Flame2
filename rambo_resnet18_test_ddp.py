import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.models import resnet18, ResNet18_Weights


# =========================
# CONFIG
# =========================
NUM_CLASSES = 5
SAMPLES = 50_000
BATCH_SIZE_PER_GPU = 256
EPOCHS = 5
IMAGE_SIZE = 224
LR = 3e-4


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def synthetic_batch(bs, device):
    x = torch.randn(
        bs, 3, IMAGE_SIZE, IMAGE_SIZE,
        device=device
    ).contiguous(memory_format=torch.channels_last)

    y = torch.randint(
        0, NUM_CLASSES,
        (bs,),
        device=device
    )
    return x, y


def run(rank, world_size):
    setup(rank, world_size)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("=" * 40)
        print(f"Running DDP on {world_size} GPUs")
        print("=" * 40)

    # -------- Model --------
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    model = model.to(device).to(memory_format=torch.channels_last)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )

    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = SAMPLES // (BATCH_SIZE_PER_GPU * world_size)

    # -------- Train --------
    for epoch in range(1, EPOCHS + 1):
        dist.barrier()
        start = time.time()

        for step in range(steps_per_epoch):
            x, y = synthetic_batch(BATCH_SIZE_PER_GPU, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        dist.barrier()
        epoch_time = time.time() - start

        if rank == 0:
            total_samples = SAMPLES
            throughput = total_samples / epoch_time
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"time {epoch_time:.2f}s | "
                f"throughput {throughput:.1f} samples/s"
            )

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
