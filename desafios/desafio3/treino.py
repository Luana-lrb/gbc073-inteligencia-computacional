# -*- coding: utf-8 -*-
"""Desafio 3 — treino configurável. Extraído das funções do notebook original.

Uma chamada = um experimento. Não executa treinos durante a importação ou --help.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

IMAGE_SIZE = 256
BATCH_SIZE = 8
GREEN_LABELS = [6, 7]
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None
LOG_EVERY_STEPS = 20
MAX_GRAD_NORM = None
BASELINE_LR = 1e-4
SWEEP_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = ""
OUTPUT_DIR = ""
NUM_WORKERS = 0 if os.name == "nt" else 2
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_samples = []
val_samples = []
test_samples = []
def normalize_name(path):
    """Remove o sufixo de cópia, como '(1)', para tentar correspondência."""
    stem = Path(path).stem.lower()
    return re.sub(r"\s*\(\d+\)$", "", stem)

def file_hash(path):
    """Calcula o hash do conteúdo do arquivo."""
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()

def collect_labeled_pairs(split_names=("Train", "Val")):
    pairs = []
    seen_images = set()
    seen_masks = set()

    stats = {
        "images_found": 0,
        "matched": 0,
        "missing_or_ambiguous": 0,
        "duplicate_images": 0,
        "duplicate_masks": 0,
    }

    for split in split_names:
        base = Path(DATASET_DIR) / split / "Urban"
        image_dir = base / "images_png"
        mask_dir = base / "masks_png"

        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Pastas não encontradas para {split}: "
                f"{image_dir} ou {mask_dir}"
            )

        masks = list(mask_dir.glob("*.png"))
        masks_exact = {p.name.lower(): p for p in masks}
        masks_normalized = defaultdict(list)

        for mask_path in masks:
            masks_normalized[normalize_name(mask_path)].append(mask_path)

        images = sorted(image_dir.glob("*.png"))
        stats["images_found"] += len(images)

        for image_path in images:
            # Primeiro tenta correspondência exata.
            mask_path = masks_exact.get(image_path.name.lower())

            # Só usa o nome normalizado se houver uma única candidata.
            if mask_path is None:
                candidates = masks_normalized.get(
                    normalize_name(image_path), []
                )
                if len(candidates) == 1:
                    mask_path = candidates[0]

            if mask_path is None:
                stats["missing_or_ambiguous"] += 1
                continue

            stats["matched"] += 1

            image_hash = file_hash(image_path)
            mask_hash = file_hash(mask_path)

            if image_hash in seen_images:
                stats["duplicate_images"] += 1
                continue

            if mask_hash in seen_masks:
                stats["duplicate_masks"] += 1
                continue

            seen_images.add(image_hash)
            seen_masks.add(mask_hash)

            pairs.append((str(image_path), str(mask_path)))

    if len(pairs) < 10:
        raise ValueError(
            f"Foram encontrados poucos pares válidos: {len(pairs)}. "
            "Confira os caminhos e os nomes das máscaras."
        )

    print("Diagnóstico do dataset:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  pares únicos aceitos: {len(pairs)}")

    return pairs

class SatelliteSegmentationDataset(Dataset):
    def __init__(self, samples, image_size=256, augment=False, green_labels=None):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.green_labels = set(GREEN_LABELS if green_labels is None else green_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        # Imagem RGB: redimensionamento bilinear mantém a aparência.
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

        # Máscara LoveDA: preservar IDs inteiros com vizinho mais próximo.
        # neste dataset, foram observados valores entre 1 e 7.
        mask_image = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask_ids = np.asarray(mask_image, dtype=np.uint8)
        mask = np.isin(mask_ids, list(self.green_labels)).astype(np.float32)

        # A mesma transformação geométrica é aplicada à imagem e à máscara.
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
            if random.random() < 0.5:
                image, mask = np.flipud(image), np.flipud(mask)
            k = random.randint(0, 3)
            if k:
                image, mask = np.rot90(image, k), np.rot90(mask, k)

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        mask = np.ascontiguousarray(mask[None, ...])
        return torch.from_numpy(image), torch.from_numpy(mask)

def build_model():
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(DEVICE)

def segmentation_metrics(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    tp = (preds * masks).sum().item()
    fp = (preds * (1 - masks)).sum().item()
    fn = ((1 - preds) * masks).sum().item()
    dice = (2*tp + 1e-8) / (2*tp + fp + fn + 1e-8)
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    def forward(self, logits, masks):
        return self.bce(logits, masks) + self.dice(logits, masks)

def zeroth_power_newton_schulz(matrix, steps=5, eps=1e-7):
    # Aproxima a parte ortogonal U @ V.T da SVD usando Newton–Schulz.
    X = matrix
    if X.ndim != 2:
        raise ValueError("A ortogonalização espera matriz 2-D.")
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transposed else X

class MuonMatrix(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.95, weight_decay=0.01, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                if group["weight_decay"]:
                    g = g.add(p, alpha=group["weight_decay"])
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                else:
                    state["momentum_buffer"].mul_(group["momentum"]).add_(g)
                mat = state["momentum_buffer"].reshape(p.shape[0], -1)
                update = zeroth_power_newton_schulz(mat, group["ns_steps"]).reshape_as(p)
                p.add_(update, alpha=-group["lr"])
        return loss

def build_optimizers(model, name, lr, adamw_lr=None):
    if name == "SGD":
        return [torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=False, weight_decay=1e-4)]
    if name == "SGD_Nesterov":
        return [torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)]
    if name == "Adam":
        return [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)]
    if name == "AdamW":
        return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)]

    if name == "Muon":
        # O Muon recebe matrizes e kernels das camadas internas.
        # A cabeça final de segmentação fica no AdamW.
        matrix_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad
            and p.ndim >= 2
            and "segmentation_head" not in n
        ]

        matrix_ids = {id(p) for p in matrix_params}

        other_params = [
            p
            for p in model.parameters()
            if p.requires_grad and id(p) not in matrix_ids
        ]

        if not matrix_params:
            raise ValueError("Não há parâmetros elegíveis para o Muon.")

        return [
            MuonMatrix(
                matrix_params,
                lr=lr,
                momentum=0.95,
                weight_decay=0.01,
                ns_steps=5,
            ),
            torch.optim.AdamW(
                other_params,
                lr=(adamw_lr or BASELINE_LR),
                weight_decay=0.01,
            ),
        ]

    raise ValueError(f"Otimizador desconhecido: {name}")

def make_schedulers(optimizers, schedule_name, total_steps, warmup_fraction=0.05):
    schedulers = []
    for opt in optimizers:
        if schedule_name == "constant":
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
        elif schedule_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=0.0)
        elif schedule_name == "warmup_cosine":
            warmup_steps = max(1, int(total_steps * warmup_fraction))
            def scale(step, w=warmup_steps, total=max(2,total_steps)):
                if step < w: return float(step + 1) / w
                progress = min(1.0, (step - w) / max(1, total - w))
                return 0.5 * (1 + math.cos(math.pi * progress))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, scale)
        elif schedule_name == "one_cycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=opt.param_groups[0]["lr"], total_steps=max(1,total_steps),
                pct_start=0.1, anneal_strategy="cos"
            )
        elif schedule_name == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1,total_steps//3), gamma=0.3)
        else:
            raise ValueError(f"Agenda desconhecida: {schedule_name}")
        schedulers.append(sched)
    return schedulers

def choose_probe_parameters(model):
    """
    Seleciona a primeira camada, uma intermediária e a última
    camada convolucional antes da cabeça de segmentação.
    """
    named = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
        and param.ndim >= 2
        and "weight" in name
        and "segmentation_head" not in name
    ]

    if len(named) < 3:
        raise ValueError(
            "Não há parâmetros matriciais suficientes "
            "para selecionar três camadas."
        )

    return {
        "primeira": named[0],
        "meio": named[len(named) // 2],
        "ultima": named[-1],
    }

def global_norm(params, use_grad=False):
    tensors = []
    for p in params:
        value = p.grad if use_grad else p
        if value is not None:
            tensors.append(value.detach().norm(2).pow(2))
    if not tensors: return 0.0
    return torch.stack(tensors).sum().sqrt().item()

def train_one_epoch(model, loader, loss_fn, optimizers, schedulers, epoch, log_rows,
                    global_step, probes, log_every=20, max_grad_norm=None):
    model.train()
    loss_sum = 0.0
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()

        grad_global = global_norm(model.parameters(), use_grad=True)  # antes de clipping
        grad_by_layer = {key: p.grad.detach().norm().item() if p.grad is not None else 0.0
                         for key, (name,p) in probes.items()}
        theta_global = global_norm(model.parameters(), use_grad=False)
        theta_by_layer = {key: p.detach().norm().item() for key, (name,p) in probes.items()}
        before = {key: p.detach().clone() for key, (name,p) in probes.items()}


        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Guardar as taxas usadas neste passo, antes de atualizar as agendas.
        lrs_used = [opt.param_groups[0]["lr"] for opt in optimizers]

        for opt in optimizers:
            opt.step()

        for sched in schedulers:
            sched.step()

        loss_sum += loss.item()
        if global_step % log_every == 0 or batch_idx == len(loader)-1:

            row = {
                "step": global_step,
                "epoch": epoch,
                "batch": batch_idx,
                "loss_train_batch": loss.item(),
                "grad_norm_global": grad_global,
                "theta_norm_global": theta_global,
                "lr": lrs_used[0],
            }

            if len(optimizers) > 1:
              row["lr_aux"] = lrs_used[1]

            for key, (name,p) in probes.items():
                row[f"grad_{key}"] = grad_by_layer[key]
                row[f"theta_{key}"] = theta_by_layer[key]
                row[f"rho_{key}"] = ((p.detach()-before[key]).norm() / (before[key].norm()+1e-12)).item()
                row[f"layer_{key}"] = name
            log_rows.append(row)
        global_step += 1
    return loss_sum / max(1,len(loader)), global_step

def evaluate(model, loader, loss_fn):
    model.eval()
    losses, all_tp, all_fp, all_fn = [], 0.0, 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        logits = model(images)
        losses.append(loss_fn(logits, masks).item())
        pred = (torch.sigmoid(logits) >= 0.5).float()
        all_tp += (pred*masks).sum().item()
        all_fp += (pred*(1-masks)).sum().item()
        all_fn += ((1-pred)*masks).sum().item()
    dice = (2*all_tp+1e-8)/(2*all_tp+all_fp+all_fn+1e-8)
    iou = (all_tp+1e-8)/(all_tp+all_fp+all_fn+1e-8)
    return {"val_loss": float(np.mean(losses)), "val_dice": dice, "val_iou": iou}

def run_experiment(optimizer_name, lr, schedule_name, seed=42, epochs=SWEEP_EPOCHS,
                   stage="sweep", adamw_lr=None, evaluate_test=False):
    set_seed(seed)
    train_ds = SatelliteSegmentationDataset(train_samples, IMAGE_SIZE, augment=True)
    val_ds = SatelliteSegmentationDataset(val_samples, IMAGE_SIZE, augment=False)
    test_ds = SatelliteSegmentationDataset(test_samples, IMAGE_SIZE, augment=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE.type=="cuda"), generator=generator)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model()
    loss_fn = SegmentationLoss()
    optimizers = build_optimizers(model, optimizer_name, lr, adamw_lr=adamw_lr)
    total_steps = max(1, epochs * len(train_loader))
    schedulers = make_schedulers(optimizers, schedule_name, total_steps)
    probes = choose_probe_parameters(model)
    log_rows, epoch_rows = [], []
    best_dice, best_state = -1.0, None
    global_step, diverged = 0, False
    start = time.time()

    print(f"\n[{stage}] {optimizer_name} | lr={lr:g} | agenda={schedule_name} | seed={seed} | épocas={epochs}")
    for epoch in range(1, epochs+1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, loss_fn, optimizers, schedulers, epoch, log_rows,
            global_step, probes, LOG_EVERY_STEPS, MAX_GRAD_NORM
        )
        val = evaluate(model, val_loader, loss_fn)
        epoch_row = {"epoch":epoch, "train_loss":train_loss, **val}
        epoch_rows.append(epoch_row)
        print(f"Época {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val['val_loss']:.4f} | Dice={val['val_dice']:.4f} | IoU={val['val_iou']:.4f}")
        if not np.isfinite(train_loss) or not np.isfinite(val["val_loss"]):
            diverged = True
            break
        if val["val_dice"] > best_dice:
            best_dice = val["val_dice"]
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    elapsed = time.time() - start
    result = {
        "stage":stage, "optimizer":optimizer_name, "lr":lr, "schedule":schedule_name,
        "seed":seed, "epochs_requested":epochs, "epochs_completed":len(epoch_rows),
        "best_val_dice":best_dice, "last_val_loss":epoch_rows[-1]["val_loss"] if epoch_rows else np.nan,
        "last_val_iou":epoch_rows[-1]["val_iou"] if epoch_rows else np.nan,
        "diverged":diverged, "seconds":elapsed, "adamw_lr":adamw_lr,
    }
    run_id = f"{stage}_{optimizer_name}_lr{lr:g}_{schedule_name}_seed{seed}".replace(".","p")
    run_dir = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(os.path.join(run_dir, f"{run_id}_signals.csv"), index=False)
    pd.DataFrame(epoch_rows).to_csv(os.path.join(run_dir, f"{run_id}_epochs.csv"), index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(run_dir, f"{run_id}_best.pt"))
    if evaluate_test and best_state is not None:
        result.update({"test_"+k:v for k,v in evaluate(model, test_loader, loss_fn).items()})
    del model, optimizers, schedulers
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

def prepare_split(dataset_dir, output_dir):
    """Carrega divisão fixa. Só cria uma nova se ainda não existir."""
    global train_samples, val_samples, test_samples
    split_path = Path(output_dir) / "split_fixo.json"
    if split_path.exists():
        with split_path.open(encoding="utf-8") as f:
            split = json.load(f)
        if split.get("seed", 42) != 42:
            print("ATENÇÃO: seed da divisão diferente de 42.")
        entries = {key: [tuple(item) for item in split[key]] for key in ("train", "val", "test")}
        # Os caminhos originais (Colab) são usados sem remapear silenciosamente.
        missing = [str(path) for arr in entries.values() for pair in arr for path in pair if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Divisão fixa existente, mas {len(missing)} caminhos não existem. Exemplo: {missing[0]}. "
                "Use o dataset no mesmo caminho dos experimentos originais."
            )
        print(f"Divisão fixa reutilizada: {split_path}")
    else:
        print("Nenhuma divisão existente; criando a divisão 70/15/15 com seed 42.")
        all_samples = collect_labeled_pairs()
        tr, remainder = train_test_split(all_samples, test_size=0.30, random_state=42, shuffle=True)
        va, te = train_test_split(remainder, test_size=0.50, random_state=42, shuffle=True)
        entries = {"train": tr, "val": va, "test": te}
        with split_path.open("w", encoding="utf-8") as f:
            json.dump({**entries, "seed": 42}, f, indent=2, ensure_ascii=False)
    train_samples, val_samples, test_samples = (entries[k] for k in ("train", "val", "test"))
    sets = [set(p[0] for p in entries[k]) for k in ("train", "val", "test")]
    if not (sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])):
        raise ValueError("Divisão contém imagens duplicadas entre treino/validação/teste")
    print(f"Amostras: treino={len(train_samples)}, validação={len(val_samples)}, teste={len(test_samples)}")


def save_result(result, output_dir, replace=False):
    stage = result["stage"]
    file = Path(output_dir) / f"{stage}.csv"
    record = pd.DataFrame([result])
    if file.exists():
        old = pd.read_csv(file)
        key = ["stage", "optimizer", "lr", "schedule", "seed"]
        same = (old[key[0]].astype(str) == str(result[key[0]]))
        for c in key[1:]:
            if c == "lr":
                same &= np.isclose(pd.to_numeric(old[c], errors="coerce"), result[c], rtol=1e-9, atol=1e-12)
            else:
                same &= old[c].astype(str) == str(result[c])
        if same.any():
            if not replace:
                raise FileExistsError("Já há resultado desta configuração e semente; use --overwrite para substituí-lo.")
            old = old.loc[~same]
        record = pd.concat([old, record], ignore_index=True)
    record.to_csv(file, index=False)
    names = ["baseline", "lr_sweep", "schedule_sweep", "confirmation", "final_test"]
    dfs = []
    for name in names:
        f = Path(output_dir) / f"{name}.csv"
        if f.exists():
            d = pd.read_csv(f)
            if len(d):
                d["arquivo_origem"] = name + ".csv"
                dfs.append(d)
    if dfs:
        pd.concat(dfs, ignore_index=True, sort=False).to_csv(Path(output_dir)/"experimentos.csv", index=False)
    print("Resultado registrado:", file)


def main():
    global DATASET_DIR, OUTPUT_DIR, NUM_WORKERS
    parser = argparse.ArgumentParser(description="Desafio 3: uma execução de treinamento U-Net sobre LoveDA")
    parser.add_argument("--optimizer", required=True, choices=["SGD", "SGD_Nesterov", "Adam", "AdamW", "Muon"])
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--schedule", default="warmup_cosine", choices=["constant", "cosine", "warmup_cosine", "one_cycle"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--stage", default="manual")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parent / "LoveDA"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "desafio3_resultados"))
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--adamw-lr", type=float, default=BASELINE_LR, help="LR do AdamW auxiliar do Muon; histórico: 1e-4")
    parser.add_argument("--evaluate-test", action="store_true", help="Somente para a etapa final")
    parser.add_argument("--overwrite", action="store_true", help="Permite sobrescrever logs/linha de uma execução existente")
    args = parser.parse_args()
    if args.lr <= 0 or args.epochs < 1 or args.adamw_lr <= 0 or args.num_workers < 0:
        parser.error("lr, adamw-lr, epochs devem ser positivos; num-workers >= 0")
    if args.evaluate_test and args.stage != "final_test":
        parser.error("--evaluate-test é permitido apenas com --stage final_test")
    DATASET_DIR, OUTPUT_DIR, NUM_WORKERS = args.dataset_dir, args.output_dir, args.num_workers
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    run_id = f"{args.stage}_{args.optimizer}_lr{args.lr:g}_{args.schedule}_seed{args.seed}".replace(".", "p")
    old_logs = list((Path(OUTPUT_DIR) / "logs").glob(run_id + "_*"))
    stage_file = Path(OUTPUT_DIR) / (args.stage + ".csv")
    old_record = False
    if stage_file.exists():
        rows = pd.read_csv(stage_file)
        if set(["optimizer", "lr", "schedule", "seed"]).issubset(rows.columns):
            old_record = bool(((rows.optimizer.astype(str) == args.optimizer)
                & np.isclose(pd.to_numeric(rows.lr, errors="coerce"), args.lr, rtol=1e-9, atol=1e-12)
                & (rows.schedule.astype(str) == args.schedule)
                & (pd.to_numeric(rows.seed, errors="coerce") == args.seed)).any())
    if (old_logs or old_record) and not args.overwrite:
        parser.error(f"Já existem logs para {run_id}. Use outro --output-dir ou --overwrite se desejar substituí-los.")
    print(f"PyTorch={torch.__version__} | SMP={smp.__version__} | dispositivo={DEVICE}")
    prepare_split(DATASET_DIR, OUTPUT_DIR)
    result = run_experiment(args.optimizer, args.lr, args.schedule, seed=args.seed,
                            epochs=args.epochs, stage=args.stage,
                            adamw_lr=args.adamw_lr if args.optimizer == "Muon" else None,
                            evaluate_test=args.evaluate_test)
    save_result(result, OUTPUT_DIR, replace=args.overwrite)
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
