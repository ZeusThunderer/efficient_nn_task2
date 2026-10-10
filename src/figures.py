"""Figures for the slides and the report. Our numbers come from report/tables, the paper's from paper_tables.json."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.paths import ROOT  # noqa: E402

TABLES = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"
PAPER = ROOT / "report" / "paper_tables.json"

COLORS = {"FT": "#c0504d", "SE": "#4f81bd", "SB": "#2e8b57", "UF": "#f79646", "BASE": "#7f7f7f"}
LABEL = {"FT": "Fine-tune", "SE": "Scratch-E", "SB": "Scratch-B", "UF": "Uniform (с нуля)"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Ours:
    def __init__(self) -> None:
        self.runs = [r for r in _read_csv(TABLES / "runs.csv") if r["test_acc"] not in ("", None)]
        self.latency = _read_csv(TABLES / "latency.csv")
        path = TABLES / "stage_keep.json"
        self.keeps = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def accs(self, cfg: str, mode: str) -> list[float]:
        return [float(r["test_acc"]) for r in self.runs if r["cfg"] == cfg and r["mode"] == mode]

    def mean(self, cfg: str, mode: str):
        values = self.accs(cfg, mode)
        return statistics.mean(values) if values else None

    def std(self, cfg: str, mode: str) -> float:
        values = self.accs(cfg, mode)
        return statistics.stdev(values) if len(values) > 1 else 0.0

    def params(self, cfg: str, mode: str):
        for r in self.runs:
            if r["cfg"] == cfg and r["mode"] == mode:
                return int(float(r["params"]))
        return None


def _paper() -> dict:
    return json.loads(PAPER.read_text(encoding="utf-8"))


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    def box(x, y, w, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.9, boxstyle="round,pad=0.05", fc=color, ec="black"))
        ax.text(x + w / 2, y + 0.45, text, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color="black"))

    ax.text(0.0, 3.15, "Классический конвейер", fontsize=11, weight="bold")
    box(0.0, 2.0, 2.6, "1. Обучить большую\nсеть", "#dce6f1")
    box(3.7, 2.0, 2.6, "2. Обрезать по\nкритерию важности", "#fde9d9")
    box(7.4, 2.0, 2.6, "3. Дообучить\n(fine-tune, 40 эп.)", "#f2dcdb")
    arrow(2.65, 2.45, 3.6, 2.45)
    arrow(6.35, 2.45, 7.3, 2.45)
    ax.text(0.0, 1.45, "Альтернатива из статьи", fontsize=11, weight="bold")
    box(3.7, 0.3, 2.6, "Та же обрезанная\nархитектура,\nслучайные веса", "#ebf1de")
    box(7.4, 0.3, 2.6, "Обучить с нуля\n(Scratch-E / Scratch-B)", "#d8e4bc")
    arrow(6.35, 0.75, 7.3, 0.75)
    arrow(5.0, 1.95, 5.0, 1.25)
    ax.text(5.15, 1.55, "берём только архитектуру", fontsize=9, style="italic")
    return _save(fig, "pipeline.png")


def predefined_vs_auto() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    base = [4, 4, 4, 4]
    pre = [2, 2, 2, 2]
    auto = [3, 4, 2, 1]
    for ax, widths, title in zip(axes, (base, pre, auto), ("Большая сеть", "Заданная (predefined):\nодна доля на слой", "Автоматическая:\nалгоритм выбирает доли")):
        for i, w in enumerate(widths):
            ax.barh(i, w, color="#4f81bd" if widths is not auto else "#2e8b57", edgecolor="black")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 4.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"слой {i + 1}" for i in range(4)])
        ax.invert_yaxis()
        ax.set_xlabel("каналов")
    return _save(fig, "predefined_vs_auto.png")


def paper_deltas() -> Path:
    paper = _paper()
    rows = []
    for r in paper["table1_l1_cifar10"]:
        rows.append((f"L1 {r['pruned']}", r))
    for r in paper["table4_slimming_cifar"][:1] + paper["table4_slimming_cifar"][5:6]:
        short = "C100" if r["dataset"] == "CIFAR-100" else "C10"
        rows.append((f"Slimming {r['model']} {r['ratio']} {short}", r))
    for r in paper["table6_unstructured_cifar10_vgg19"]:
        rows.append((f"Han VGG-19 {r['ratio']}", r))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = range(len(rows))
    width = 0.38
    ax.bar([i - width / 2 for i in x], [r["se"] - r["ft"] for _, r in rows], width, color=COLORS["SE"], label="Scratch-E − Fine-tune")
    ax.bar([i + width / 2 for i in x], [r["sb"] - r["ft"] for _, r in rows], width, color=COLORS["SB"], label="Scratch-B − Fine-tune")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([name for name, _ in rows], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("разница точности, п.п.")
    ax.set_title("Таблицы 1, 4, 6 статьи: обучение с нуля относительно fine-tune")
    ax.legend(fontsize=8)
    return _save(fig, "paper_deltas.png")


def paper_arch() -> Path:
    data = _paper()["table7_vgg16_slimming_60"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = range(1, len(data["width"]) + 1)
    ax.bar([i - 0.2 for i in x], data["width"], 0.4, color="#bfbfbf", edgecolor="black", label="исходная VGG-16")
    ax.bar([i + 0.2 for i in x], data["width_pruned"], 0.4, color=COLORS["SB"], edgecolor="black", label="после Slimming 60%")
    ax.set_xlabel("номер свёрточного слоя")
    ax.set_ylabel("каналов")
    ax.set_title("Таблица 7 статьи: найденная архитектура режет в основном поздние слои")
    ax.legend(fontsize=8)
    return _save(fig, "paper_arch.png")


def ours_h1(ours: Ours) -> Path:
    cfgs = ["L1-V16A", "L1-R56A", "L1-R56B", "SL-V19-70", "SL-R56-60"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.26
    for k, mode in enumerate(("FT", "SE", "SB")):
        xs = [i + (k - 1) * width for i in range(len(cfgs))]
        ys = [ours.mean(c, mode) or 0 for c in cfgs]
        es = [ours.std(c, mode) for c in cfgs]
        ax.bar(xs, ys, width, yerr=es, capsize=2, color=COLORS[mode], label=LABEL[mode])
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([c + ("*" if c.startswith("L1") else "") for c in cfgs], fontsize=9)
    ax.set_ylim(89.5, 94)
    ax.set_ylabel("точность на тесте, %")
    ax.set_title("Наши результаты, 5 основных конфигов (среднее и std по 2 сидам)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    return _save(fig, "ours_h1.png")


def ours_slim_resnet(ours: Ours) -> Path:
    points = ["40", "60", "80"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for mode, cfg_prefix in (("FT", "SL-R56-"), ("SE", "SL-R56-"), ("SB", "SL-R56-"), ("UF", "UF-R56-")):
        ys = [ours.mean(cfg_prefix + p, mode) for p in points]
        ax.plot([int(p) for p in points], ys, marker="o", color=COLORS[mode], label=LABEL[mode])
    ax.set_xlabel("доля обрезанных внутренних каналов, %")
    ax.set_ylabel("точность на тесте, %")
    ax.set_title("ResNet-56, Network Slimming: разрыв растёт с прунингом")
    ax.set_xticks([40, 60, 80])
    ax.legend(fontsize=8)
    return _save(fig, "ours_slim_resnet.png")


def ours_vgg_uf(ours: Ours) -> Path:
    points = ["30", "50", "70"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sl_p, sl_a, uf_p, uf_a = [], [], [], []
    for p in points:
        sl_cfg = f"SL-V19-{p}"
        uf_cfg = f"UF-V19-{p}"
        s_acc = ours.mean(sl_cfg, "SE")
        u_acc = ours.mean(uf_cfg, "UF")
        s_par = ours.params(sl_cfg, "SE")
        u_par = ours.params(uf_cfg, "UF")
        if None in (s_acc, u_acc, s_par, u_par):
            continue
        sl_p.append(s_par / 1e6)
        sl_a.append(s_acc)
        uf_p.append(u_par / 1e6)
        uf_a.append(u_acc)
    ax.plot(sl_p, sl_a, marker="o", color=COLORS["SE"], label="архитектура Slimming (с нуля)")
    ax.plot(uf_p, uf_a, marker="s", color=COLORS["UF"], label="uniform (с нуля)")
    ax.set_xlabel("параметров, млн")
    ax.set_ylabel("точность на тесте, %")
    ax.set_title("VGG-19, CIFAR-10: аналог рис. 3 статьи")
    ax.legend(fontsize=8)
    return _save(fig, "ours_vgg_uf.png")


def ours_stage(ours: Ours) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    wanted = [("SL-V19-30", "30%"), ("SL-V19-50", "50%"), ("SL-V19-70", "70%")]
    for cfg, label in wanted:
        row = next((r for r in ours.keeps if r["cfg"] == cfg), None)
        if row is None:
            continue
        rates = [float(v) for v in row["rates"].split()]
        ax.plot(range(1, len(rates) + 1), rates, marker="o", label=f"Slimming {label}")
    ax.set_xlabel("стадия VGG-19 (по размеру карты признаков)")
    ax.set_ylabel("доля сохранённых каналов")
    ax.set_title("Найденные архитектуры VGG-19: поздние стадии режутся сильнее")
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=8)
    return _save(fig, "ours_stage.png")


def ours_unstructured(ours: Ours) -> Path:
    cfgs = ["UN-V19-80", "UN-V19-95", "UN-R56-95"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.26
    for k, mode in enumerate(("FT", "SE", "SB")):
        xs = [i + (k - 1) * width for i in range(len(cfgs))]
        ys = [ours.mean(c, mode) or 0 for c in cfgs]
        ax.bar(xs, ys, width, color=COLORS[mode], label=LABEL[mode])
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, fontsize=9)
    ax.set_ylim(85, 94.5)
    ax.set_ylabel("точность на тесте, %")
    ax.set_title("Нерегулярный прунинг (сид 1)")
    ax.legend(fontsize=8)
    return _save(fig, "ours_unstructured.png")


def ours_latency(ours: Ours) -> Path:
    rows = [r for r in ours.latency if r["dtype"] == "fp32"]
    names = []
    for r in rows:
        if r["run_id"] not in names:
            names.append(r["run_id"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, batch in zip(axes, ("1", "256")):
        values = [float(next(r["median_ms"] for r in rows if r["run_id"] == n and r["batch"] == batch)) for n in names]
        ax.barh(range(len(names)), values, color="#4f81bd")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace(".SE.s1", "").replace(".BASE.s1", " (база)") for n in names], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("мс на батч, FP32")
        ax.set_title(f"batch {batch}")
    return _save(fig, "ours_latency.png")


def make_all() -> dict[str, Path]:
    ours = Ours()
    out = {
        "pipeline": pipeline(),
        "predefined_vs_auto": predefined_vs_auto(),
        "paper_deltas": paper_deltas(),
        "paper_arch": paper_arch(),
        "ours_h1": ours_h1(ours),
        "ours_slim_resnet": ours_slim_resnet(ours),
        "ours_vgg_uf": ours_vgg_uf(ours),
        "ours_stage": ours_stage(ours),
        "ours_unstructured": ours_unstructured(ours),
        "ours_latency": ours_latency(ours),
    }
    return out


if __name__ == "__main__":
    for name, path in make_all().items():
        print(name, path)
