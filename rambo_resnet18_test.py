import time
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# =========================
# CONFIG
# =========================
NUM_CLASSES = 5
SAMPLES = 50_000
BATCH_SIZE = 1028      # תשנה ל-128 / 256 / 384 / 512 לבדיקה
EPOCHS = 5
IMAGE_SIZE = 224
LR = 3e-4

# =========================
# PERFORMANCE SETTINGS
# =========================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 36)
print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 36)

# =========================
# MODEL (Transfer Learning)
# =========================
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# replace classifier
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# freeze all
for p in model.parameters():
    p.requires_grad = False

# unfreeze last layers
for p in model.layer3.parameters():
    p.requires_grad = True
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

model = model.to(device).to(memory_format=torch.channels_last)

# =========================
# OPTIMIZER + AMP
# =========================
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR
)

scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()

# =========================
# SYNTHETIC DATA
# =========================
def synthetic_batch(bs):
    x = torch.randn(
        bs, 3, IMAGE_SIZE, IMAGE_SIZE,
        device=device,
        dtype=torch.float32
    ).contiguous(memory_format=torch.channels_last)

    y = torch.randint(
        0, NUM_CLASSES,
        (bs,),
        device=device
    )
    return x, y


# =========================
# TRAIN LOOP
# =========================
print("Starting synthetic training test...")
print(f"Samples: {SAMPLES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("-" * 36)

total_start = time.time()
steps_per_epoch = SAMPLES // BATCH_SIZE

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_start = time.time()

    for step in range(steps_per_epoch):
        x, y = synthetic_batch(BATCH_SIZE)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            elapsed = (time.time() - epoch_start) / 60
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {step:4d}/{steps_per_epoch} | "
                f"Loss {loss.item():.4f} | "
                f"Elapsed {elapsed:.1f} min"
            )

    epoch_time = (time.time() - epoch_start) / 60
    print(f"Epoch {epoch} finished in {epoch_time:.2f} minutes")

total_time = (time.time() - total_start) / 60
print("=" * 36)
print("Training test completed successfully")
print(f"Total runtime: {total_time:.2f} minutes")
print("=" * 36)
