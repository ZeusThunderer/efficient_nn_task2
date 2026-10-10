"""Two documents in Russian.

report/slides.pdf  - a talk that explains the paper (arXiv:1810.05270).
report/report.pdf  - a report on our reproduction of the paper's results.

Our numbers come from report/tables/*.csv|json (written by src.aggregate and scripts/latency.py).
The paper's numbers come from report/paper_tables.json (transcribed from the PDF).
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from fpdf import FPDF
from PIL import Image

from src.aggregate import build
from src.figures import Ours, make_all
from src.paths import FROZEN_HPARAMS, ROOT

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_B = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
TABLES = ROOT / "report" / "tables"
PAPER = ROOT / "report" / "paper_tables.json"

DARK = (31, 56, 100)
LIGHT = (234, 239, 247)


# ---------------------------------------------------------------- data

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load() -> dict:
    build(TABLES)
    figs = make_all()
    plans = {}
    for name in ("f0", "plan_f0", "plan_f1", "plan_f2"):
        plans[name] = _json(ROOT / "budget" / f"{name}.json", {})
    dev = [
        row
        for row in _read_csv(ROOT / "dev" / "summary.csv")
        if str(row.get("run_id", "")).startswith("dev-") and row.get("val_acc")
    ]
    return {
        "paper": _json(PAPER, {}),
        "ours": Ours(),
        "hyp": _json(TABLES / "hypotheses.json", {}),
        "inv": _json(TABLES / "invariants.json", {"ok": False, "errors": []}),
        "frozen": _json(FROZEN_HPARAMS, {}),
        "plans": plans,
        "dev": dev,
        "summary": _read_csv(TABLES / "summary.csv"),
        "runs": _read_csv(TABLES / "runs.csv"),
        "figs": figs,
    }


def _ms(ours: Ours, cfg: str, mode: str) -> str:
    values = ours.accs(cfg, mode)
    if not values:
        return "нет"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def _delta(ours: Ours, cfg: str, a: str, b: str) -> str:
    x, y = ours.mean(cfg, a), ours.mean(cfg, b)
    if x is None or y is None:
        return "нет"
    return f"{x - y:+.2f}"


def _n(ours: Ours, cfg: str, mode: str) -> int:
    return len(ours.accs(cfg, mode))


def _macs_of(ours: Ours, cfg: str) -> int | None:
    for r in ours.runs:
        if r["cfg"] == cfg:
            return int(float(r["macs"]))
    return None


def _macs_cut(ours: Ours, cfg: str, base: str) -> str:
    small, large = _macs_of(ours, cfg), _macs_of(ours, base)
    if small is None or large is None:
        return "нет"
    return f"−{(1 - small / large) * 100:.1f}"


def _capped(data: dict) -> str:
    """Configs whose F_L / F_P exceeds the 2x cap on Scratch-B epochs (from PRUNE rows of runs.csv)."""
    ours: Ours = data["ours"]
    base_macs = {}
    for r in data["runs"]:
        if r["mode"] == "BASE" and r["seed"] == "1":
            base_macs[r["cfg"]] = float(r["macs"])
    seen: dict[str, list[str]] = {}
    for r in data["runs"]:
        if r["mode"] != "PRUNE":
            continue
        base = "V19" if "V19" in r["cfg"] else "V16" if "V16" in r["cfg"] else "R56"
        ratio = base_macs[base] / float(r["macs"])
        if ratio > 2:
            seen.setdefault(r["cfg"], []).append(f"{ratio:.2f} (сид {r['seed']})")
    return "; ".join(f"{cfg}: F_L/F_P = " + ", ".join(v) for cfg, v in seen.items())


def _lat_fp16(lat: list[dict], ours: Ours) -> str:
    def v(run: str, batch: str, dtype: str = "fp16") -> float:
        return float(next(r["median_ms"] for r in lat if r["run_id"] == run and r["batch"] == batch and r["dtype"] == dtype))

    r56 = ["L1-R56A", "L1-R56B", "SL-R56-40", "SL-R56-60", "SL-R56-80"]
    v19 = ["SL-V19-30", "SL-V19-50", "SL-V19-70"]
    lo256, hi256 = min(v(f"{c}.SE.s1", "256") for c in r56), max(v(f"{c}.SE.s1", "256") for c in r56)
    lo1, hi1 = min(v(f"{c}.SE.s1", "1") for c in r56), max(v(f"{c}.SE.s1", "1") for c in r56)
    base256, base1 = v("R56.BASE.s1", "256"), v("R56.BASE.s1", "1")
    assert lo256 > base256 and lo1 > base1
    sl30, v19_256 = v("SL-V19-30.SE.s1", "256"), v("V19.BASE.s1", "256")
    assert sl30 > v19_256
    lo19, hi19 = min(v(f"{c}.SE.s1", "1") for c in v19), max(v(f"{c}.SE.s1", "1") for c in v19)
    base19_1 = v("V19.BASE.s1", "1")
    assert lo19 > base19_1
    return (
        "FP16 (медианы из таблицы): ни одна обрезанная ResNet-56 не быстрее базы. На batch 256 они занимают "
        f"{lo256:.2f}–{hi256:.2f} мс против {base256:.2f} у базы, на batch 1 — {lo1:.2f}–{hi1:.2f} против {base1:.2f}. "
        f"SL-V19-30 на batch 256 даёт {sl30:.2f} мс против {v19_256:.2f} у V19 при {_macs_cut(ours, 'SL-V19-30', 'V19')}% MAC. "
        f"На batch 1 все SL-V19 ({lo19:.2f}–{hi19:.2f} мс) медленнее базы ({base19_1:.2f} мс). "
        "Вывод про ускорение прунингом поэтому справедлив для FP32, но не для FP16. "
        "Гипотеза, не проверялась: ширины после прунинга не кратны 8 и выпадают из tensor-core путей. "
        "Проверка (округлить ширины вверх до кратных 8 и перемерить) не проводилась."
    )


def _signed(value) -> str:
    if value is None or value == "":
        return "нет"
    return f"{float(value):+.2f}"


# ---------------------------------------------------------------- pdf helpers

class Doc(FPDF):
    def __init__(self, landscape: bool = False):
        super().__init__(orientation="L" if landscape else "P", unit="mm", format="A4")
        self.add_font("DejaVu", "", str(FONT))
        self.add_font("DejaVu", "B", str(FONT_B))
        self.set_auto_page_break(auto=True, margin=14)
        self.set_margins(16, 14, 16)

    def text_block(self, text: str, size: float = 10.5, bold: bool = False, gap: float = 1.5, line: float | None = None) -> None:
        self.set_font("DejaVu", "B" if bold else "", size)
        self.multi_cell(0, line or size * 0.55, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(gap)

    def table(self, rows: list[list[str]], widths: list[float], size: float = 8.5, header_fill=LIGHT, align: str = "CENTER"):
        self.set_font("DejaVu", "", size)
        with super().table(
            col_widths=widths,
            text_align=align,
            line_height=size * 0.62,
            first_row_as_headings=True,
            headings_style=__import__("fpdf").FontFace(emphasis="BOLD", fill_color=header_fill),
        ) as tbl:
            for row in rows:
                cells = tbl.row()
                for value in row:
                    cells.cell(str(value))
        self.ln(2)

    def picture(self, path: Path, width: float) -> None:
        with Image.open(path) as img:
            ratio = img.height / img.width
        height = width * ratio
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page()
        x = (self.w - width) / 2
        self.image(str(path), x=x, y=self.get_y(), w=width)
        self.set_y(self.get_y() + height + 2)


class Slides(Doc):
    def __init__(self):
        super().__init__(landscape=True)
        self.set_auto_page_break(auto=False)
        self.set_margins(14, 12, 14)

    def slide(self, title: str, bullets: list[str] | None = None, image: Path | None = None, image_w: float = 150,
              layout: str = "below", table=None, note: str | None = None, size: float = 15) -> None:
        self.add_page()
        self.set_fill_color(*DARK)
        self.rect(0, 0, self.w, 20, style="F")
        self.set_text_color(255, 255, 255)
        self.set_xy(14, 5)
        self.set_font("DejaVu", "B", 20)
        self.cell(0, 10, title)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)
        size = size * 1.2
        y0 = 28
        self.set_xy(14, y0)
        text_w = self.w - 28
        if image is not None and layout == "right":
            text_w = self.w - 28 - image_w - 6
        if bullets:
            self.set_font("DejaVu", "", size)
            for item in bullets:
                self.set_x(14)
                self.multi_cell(text_w, size * 0.5, "• " + item, new_x="LMARGIN", new_y="NEXT", align="L")
                self.ln(1.8)
        if table is not None:
            rows, widths, tsize = table
            k = min(1.35, (self.w - 28) / sum(widths))
            self.set_x(14)
            self.table(rows, [w * k for w in widths], size=tsize * 1.3)
        if image is not None:
            with Image.open(image) as img:
                ratio = img.height / img.width
            h = image_w * ratio
            if layout == "right":
                self.image(str(image), x=self.w - 14 - image_w, y=y0, w=image_w)
            else:
                y = max(self.get_y() + 2, y0)
                limit = self.h - 16 - y
                if h > limit:
                    image_w = image_w * limit / h
                    h = limit
                self.image(str(image), x=(self.w - image_w) / 2, y=y, w=image_w)
        if note:
            self.set_xy(14, self.h - 14)
            self.set_font("DejaVu", "", 9)
            self.set_text_color(90, 90, 90)
            self.cell(0, 6, note)
            self.set_text_color(0, 0, 0)


# ---------------------------------------------------------------- slides

def _paper_rows(rows: list[dict], labeler, unpruned_key: str = "unpruned") -> list[list[str]]:
    out = [["Модель", "Исходная", "Fine-tune", "Scratch-E", "Scratch-B"]]
    for r in rows:
        out.append([labeler(r), f"{r[unpruned_key]:.2f}", f"{r['ft']:.2f}", f"{r['se']:.2f}", f"{r['sb']:.2f}"])
    return out


def write_slides(data: dict, path: Path) -> None:
    paper = data["paper"]
    ours: Ours = data["ours"]
    figs = data["figs"]
    hyp = data["hyp"]
    s = Slides()

    s.slide(
        "Rethinking the Value of Network Pruning",
        [
            "Liu, Sun, Zhou, Huang, Darrell. ICLR 2019, arXiv:1810.05270",
            "Разбор статьи: что утверждают авторы, как проверяют и что из этого следует",
            "В конце: что получилось при моём воспроизведении на CIFAR",
        ],
        size=18,
    )

    s.slide(
        "О чём статья",
        [
            "Вопрос: нужны ли в прунинге сами унаследованные веса большой сети?",
            "Привычный ответ: да. Обучить большую сеть, убрать «ненужное», дообучить остаток.",
            "Ответ статьи для структурированного прунинга: нет. Та же обрезанная архитектура, обученная с нуля из случайных весов, даёт не хуже.",
            "Следствие: ценность прунинга — в найденной архитектуре (число каналов в слоях), а не в весах.",
            "Исключение, которое авторы честно указывают: нерегулярный (поэлементный) прунинг на ImageNet и очень высокой разреженности.",
        ],
    )

    s.slide(
        "Классический конвейер и две общие веры",
        [
            "Вера 1: сначала нужна большая перепараметризованная сеть, из неё безопасно убирать лишнее, и это лучше обучения малой сети с нуля.",
            "Вера 2: важны и архитектура, и веса, которые «пережили» прунинг. Поэтому почти все методы дообучают, а не обучают заново.",
        ],
        image=figs["pipeline"],
        image_w=190,
        size=14,
    )

    s.slide(
        "Какие бывают целевые архитектуры",
        [
            "Структурированный прунинг убирает целые каналы или блоки. Остаётся обычная плотная свёртка, спецбиблиотеки не нужны.",
            "Нерегулярный прунинг (Han et al.) обнуляет отдельные веса. Матрицы разрежены, ускорение без специального железа не получить.",
            "Predefined: человек задаёт долю каналов на слой (L1-norm, ThiNet, Regression).",
            "Automatic: алгоритм глобально сравнивает важность и сам выбирает ширину слоёв (Network Slimming, SSS).",
        ],
        image=figs["predefined_vs_auto"],
        image_w=150,
        size=13,
    )

    s.slide(
        "Как честно сравнивать с обучением с нуля",
        [
            "Scratch-E: малая сеть учится столько же эпох, что и большая (160 на CIFAR). Это недоучёт: эпоха малой сети дешевле.",
            "Scratch-B: эпох столько, чтобы совпали FLOPs с обучением большой сети. График lr растягивается пропорционально. На ImageNet при ускорении больше 2× эпохи просто удваиваются.",
            "Стоимость fine-tune в бюджет не включают. Авторы отмечают, что это в пользу конвейера прунинга.",
            "Протокол: SGD с Нестеровым, ступенчатый lr, He-инициализация. CIFAR: 160 эпох обучения, 40 дообучения. ImageNet: 90 и 20. 5 сидов, среднее и std.",
            "Fine-tune идёт с самым маленьким lr большой сети. На ImageNet сравнивают относительное падение точности из-за разницы фреймворков (PyTorch против Caffe).",
        ],
        size=14,
    )

    s.slide(
        "Что проверяли",
        [
            "Predefined структурированные: L1-norm filter pruning (Li et al.), ThiNet (Luo et al.), Regression-based reconstruction (He et al.), Soft Filter Pruning (приложение).",
            "Automatic структурированные: Network Slimming (Liu et al.), Sparse Structure Selection (Huang, Wang).",
            "Нерегулярный: magnitude pruning (Han et al.), только веса свёрток.",
            "Данные: CIFAR-10, CIFAR-100, ImageNet. Сети: VGG, ResNet, DenseNet, PreResNet. Отдельно перенос на детекцию (приложение B).",
            "Запущено: L1-norm, Network Slimming, Han, uniform. Планировал, не успел: guided pruning, Lottery Ticket, CIFAR-100.",
        ],
        size=14,
    )

    t1 = _paper_rows(paper["table1_l1_cifar10"], lambda r: r["pruned"])
    s.slide(
        "Результат 1: L1-norm, CIFAR-10 (таблица 1 статьи)",
        [
            "Scratch-E в пределах 0.15 п.п. от fine-tune или выше, Scratch-B выше fine-tune во всех трёх строках (на 0.1–0.4 п.п.).",
            "На ImageNet (ResNet-34) оба Scratch-B лучше fine-tune заметно.",
        ],
        table=(t1, [58, 34, 34, 34, 34], 11),
        note="Точность на тесте, %. Среднее по 5 сидам. ResNet-110 из таблицы 1 не приведён.",
        size=14,
    )

    thinet = [["Модель", "Fine-tune", "Scratch-E", "Scratch-B"]]
    for r in paper["table2_thinet_imagenet_rel_drop"]:
        thinet.append([f"ThiNet {r['pruned']}", f"{r['ft']:+.2f}", f"{r['se']:+.2f}", f"{r['sb']:+.2f}"])
    for r in paper["table3_regression_imagenet_rel_drop"]:
        thinet.append([f"Regression {r['pruned']}", f"{r['ft']:+.2f}", f"{r['se']:+.2f}", f"{r['sb']:+.2f}"])
    s.slide(
        "Результат 2: ThiNet и Regression на ImageNet",
        [
            "Показано относительное падение точности от большой сети (таблицы 2 и 3).",
            "Scratch-B не хуже fine-tune во всех строках. Scratch-E лучше у ResNet-50 и хуже у VGG-16 (в VGG-Tiny бюджет срезан в разы).",
        ],
        table=(thinet, [70, 40, 40, 40], 10),
        size=14,
    )

    t4 = _paper_rows(
        paper["table4_slimming_cifar"],
        lambda r: f"{r['dataset']} {r['model']} {r['ratio']}",
    )
    s.slide(
        "Результат 3: Network Slimming (таблица 4)",
        [
            "Automatic прунинг: Scratch-B лучше fine-tune в 8 из 10 экспериментов, Scratch-E слегка хуже, но чаще внутри std.",
        ],
        table=(t4, [74, 30, 30, 30, 30], 9.5),
        note="Точность на тесте, %, среднее по 5 сидам.",
        size=14,
    )

    t6 = _paper_rows(paper["table6_unstructured_cifar10_vgg19"], lambda r: f"VGG-19 CIFAR-10 {r['ratio']}")
    t6 += _paper_rows(paper["table6_unstructured_cifar10_preresnet110"], lambda r: f"PreResNet-110 CIFAR-10 {r['ratio']}")[1:]
    s.slide(
        "Результат 4: нерегулярный прунинг (таблица 6)",
        [
            "На CIFAR при доле до 80% Scratch-B не хуже fine-tune. При 95% fine-tune иногда лучше (PreResNet-110).",
            "На ImageNet Scratch-B в основном заметно хуже fine-tune. Авторы связывают это со сложностью обучения очень разреженной сети и с тем, что нерегулярный прунинг сильно меняет распределение весов.",
        ],
        table=(t6, [74, 30, 30, 30, 30], 9.5),
        size=13,
    )

    s.slide(
        "Сводка: обучение с нуля против fine-tune",
        [
            "Положительная разница Scratch-B означает, что обучение с нуля выигрывает у fine-tune.",
            "Scratch-B−FT в основном положительная, Scratch-E−FT знакопеременная; величины порядка десятых долей п.п.",
        ],
        image=figs["paper_deltas"],
        image_w=190,
        note="Построено по таблицам 1, 4 и 6 статьи, CIFAR.",
        size=14,
    )

    s.slide(
        "Прунинг как поиск архитектуры",
        [
            "Если веса не важны, то важна форма сети. Проверка: обучить с нуля архитектуру из прунинга и сравнить с uniform-прунингом (одна доля на все слои).",
            "VGG-16 на CIFAR-10 с Network Slimming: найденные архитектуры достигают той же точности с примерно в 5 раз меньшим числом параметров, чем uniform (рис. 3).",
            "Найденные ширины стабильны между сидами и систематичны: почти не режутся ранние слои, сильно режутся поздние (таблица 7).",
        ],
        image=figs["paper_arch"],
        image_w=165,
        size=13,
    )

    s.slide(
        "Когда архитектура не лучше uniform. Guided и перенос",
        [
            "У PreResNet-164 (Slimming) и у PreResNet-110, DenseNet-40 (нерегулярный прунинг) найденные архитектуры почти равномерны по стадиям (таблицы 18–20). Там прунинг не лучше uniform (рис. 5).",
            "У VGG избыточность неравномерна (таблица 21), и найденная архитектура всегда лучше uniform.",
            "Guided pruning: взять среднее число каналов по стадиям из найденных архитектур и построить из этого сеть. Она не уступает найденной (рис. 6).",
            "Transferred guided: шаблон из VGG-16 на CIFAR-10 переносится на VGG-19 на CIFAR-100 и всё равно заметно лучше uniform. Большую сеть для такого дизайна учить не нужно.",
        ],
        size=14,
    )

    lth = [["Доля", "Winning ticket, lr 0.1", "Случайная инициализация, lr 0.1", "Winning ticket, lr 0.01", "Случайная инициализация, lr 0.01"]]
    for a, b in zip(paper["table9_lth_one_shot_lr01_vgg16"], paper["table9_lth_one_shot_lr001_vgg16"]):
        lth.append([a["ratio"], f"{a['winning']:.2f}", f"{a['random']:.2f}", f"{b['winning']:.2f}", f"{b['random']:.2f}"])
    s.slide(
        "Lottery Ticket Hypothesis (раздел 6)",
        [
            "Гипотеза: подсеть с исходной инициализацией («выигрышный билет») обучается лучше, чем та же подсеть со случайными весами.",
            "Авторы: при lr 0.1, стандартном для CIFAR и ImageNet, выигрыша нет. Он появляется только при малом lr 0.01, но тогда итоговая точность ниже.",
            "Для структурированного прунинга (L1-norm) билет не помогает ни при каком lr (таблица 8).",
        ],
        table=(lth, [28, 52, 62, 52, 62], 9),
        note="VGG-16, CIFAR-10, одношаговый нерегулярный прунинг, таблица 9.",
        size=13,
    )

    s.slide(
        "Выводы статьи и практика",
        [
            "Для структурированного прунинга сильная базовая линия — обучить целевую архитектуру с нуля. Любое новое сравнение должно её включать.",
            "Обучение с нуля проще: не нужны критерий прунинга, настройка его гиперпараметров и этап дообучения; модель меньше, значит и память меньше.",
            "Fine-tune всё равно удобен, если большая обученная модель уже есть, а бюджета на обучение нет, или если нужно сразу несколько размеров.",
            "Нерегулярный прунинг на большом масштабе ведёт себя иначе: веса там могут быть важнее.",
        ],
        size=15,
    )

    s.slide(
        "Что слабо в статье",
        [
            "Часть целевых архитектур в оригинальных работах задана неполно, поэтому их приходится восстанавливать.",
            "Бюджет Scratch-B равен обучению большой сети без fine-tune, то есть меньше полного конвейера (обучение + fine-tune). Авторы называют это уступкой в пользу прунинга.",
            "Возражение «fine-tune 40 эпох мало» снято в приложении D статьи: в таблице 16 fine-tune 80 и 160 эпох заметного прироста не дают.",
            "Результаты на CIFAR отличаются на десятые доли п.п., а std такого же порядка. Выводы статистические, а не «железные».",
            "ImageNet показывает исключение для нерегулярного прунинга, но не объясняет его до конца.",
            "Это моя оценка, а не цитата из статьи.",
        ],
        size=14,
    )

    # --- our reproduction
    s.slide(
        "Моё воспроизведение: что я сделал",
        [
            "Одна RTX 5070 Ti, один общий цикл обучения для всех режимов, гиперпараметры заморожены на dev-сплите до запуска тестов.",
            "Сети: VGG-16/19 и ResNet-56 на CIFAR-10. L1-norm, Network Slimming, Han, uniform. Три конфига L1 (VGG-16-A, ResNet-56-A/B) в статье заданы неполно, я восстановил их сам и пометил звёздочкой.",
            "Основные 5 конфигов прошли по 2 сида, остальные по 1. Статья: 5 сидов.",
            "Не успел по бюджету GPU: сид 3, CIFAR-100, guided pruning, winning ticket, Scratch-B без потолка 2×.",
        ],
        size=14,
    )

    s.slide(
        "Воспроизведение: Scratch-B не хуже fine-tune",
        [
            f"Критерий H1 (SB−FT ≥ −0.2 п.п. в 4 из 5 и SE−FT ≥ −0.3 в 3 из 5): {hyp.get('H1', 'нет данных')}.",
            "Хуже всего L1-R56A*: SB−FT = "
            + _delta(ours, "L1-R56A", "SB", "FT")
            + " п.п. Лучше всего SL-R56-60: SB−FT = "
            + _delta(ours, "SL-R56-60", "SB", "FT")
            + " п.п.",
            "n=2, std 0.01–0.87 п.п. (по 5 конфигам и трём режимам) сопоставим с порогами критерия, поэтому «подтверждена» читать как «согласуется со статьёй». Ось Y на графике обрезана.",
        ],
        image=figs["ours_h1"],
        image_w=150,
        size=14,
    )

    sl_uf_res = ", ".join(f"{r['point']}%: {r['abs_sl_minus_uf']:+.2f}" for r in hyp.get("H3_resnet_rows", []))
    s.slide(
        "Воспроизведение: архитектура против uniform",
        [
            f"VGG-19, критерий H3: {hyp.get('H3_vgg', 'нет данных')}. SL−UF (сид 1) = " + ", ".join(f"{r['sl_minus_uf']:+.2f}" for r in hyp.get("H3_vgg_rows", [])) + " п.п. при 30/50/70%. Это аналог рис. 3 на VGG-19 (в статье рис. 3 построен на VGG-16).",
            "Поздние стадии режутся сильнее, как в таблице 21 статьи (VGG-16).",
            "В статье паритет со Slimming виден на PreResNet-164 (рис. 5 слева, таблица 18); Slimming на ResNet-56 там нет. Паритет нерегулярного прунинга с uniform — на PreResNet-110 и DenseNet-40 (рис. 5 справа и в центре).",
            f"У меня на ResNet-56 разрыв SL−UF остаётся (сид 1; 40/60/80%: {sl_uf_res}): ширины неравномерны, размах по стадиям 0.37–0.39. Вердикт H3 для ResNet: {hyp.get('H3_resnet', 'нет данных')}.",
        ],
        image=figs["ours_vgg_uf"],
        image_w=95,
        layout="right",
        size=12,
    )

    s.slide(
        "Воспроизведение: что не совпало со статьёй",
        [
            "Абсолютная точность базовых VGG на тесте ниже статьи на 0.7–0.8 п.п., у ResNet-56 на 0.1 п.п.; причину не установил. Сравнение внутри моего цикла остаётся честным.",
            f"H4 {hyp.get('H4', 'нет данных')}: на ResNet-56 при 95% нерегулярный Scratch-B выше fine-tune на {hyp.get('H4_un95_sb_minus_ft', 0):+.2f} п.п. H4 опиралась на перенос поведения PreResNet-110 на ResNet-56; это моё допущение, а не результат статьи.",
            "Предполагаю (не проверял): fine-tune 40 эпох на сильно разреженной сети недообучается.",
            "Winning ticket и guided pruning в итог не вошли: эти прогоны вырезаны бюджетом.",
        ],
        size=14,
    )

    s.slide(
        "Итоги",
        [
            "Главный тезис для структурированного прунинга воспроизвёлся на CIFAR-10: обучение с нуля при бюджете Scratch-B не хуже fine-tune, а чаще лучше.",
            "Архитектура, найденная Slimming на VGG-19, лучше uniform при том же числе параметров.",
            f"Исключение для нерегулярного прунинга в статье — ImageNet и 95% на PreResNet-110 и DenseNet-BC-100; их я не проверял. На VGG-19 95% и у статьи, и у меня Scratch-B ≥ fine-tune. Мой аналог на ResNet-56 (H4) опровергнут: SB−FT {hyp.get('H4_un95_sb_minus_ft', 0):+.2f} п.п.",
            "Полный отчёт со всеми таблицами, отклонениями и ограничениями: report/report.pdf.",
        ],
        size=15,
    )
    s.output(str(path))


# ---------------------------------------------------------------- report

def _comparison_table(ours: Ours, paper_rows: list[dict], our_rows: list[tuple[str, str, str]], label_paper, with_macs: bool = False) -> list[list[str]]:
    """our_rows: (display label, cfg, base cfg)."""
    rows = [["Строка", "Исходная", "Fine-tune", "Scratch-E", "Scratch-B", "SE−FT", "SB−FT"] + (["−MACs, %"] if with_macs else [])]
    for pr in paper_rows:
        rows.append([
            label_paper(pr),
            f"{pr['unpruned']:.2f}",
            f"{pr['ft']:.2f}",
            f"{pr['se']:.2f}",
            f"{pr['sb']:.2f}",
            f"{pr['se'] - pr['ft']:+.2f}",
            f"{pr['sb'] - pr['ft']:+.2f}",
        ] + (["—"] if with_macs else []))
    for label, cfg, base in our_rows:
        base_mode = "BASES" if cfg.startswith("SL") else "BASE"
        rows.append([
            label,
            _ms(ours, base, base_mode),
            _ms(ours, cfg, "FT"),
            _ms(ours, cfg, "SE"),
            _ms(ours, cfg, "SB"),
            _delta(ours, cfg, "SE", "FT"),
            _delta(ours, cfg, "SB", "FT"),
        ] + ([_macs_cut(ours, cfg, base)] if with_macs else []))
    return rows


FOOTNOTE = "Разности считаются по нескруглённым значениям; в таблицах показаны округлённые средние."


def write_report(data: dict, path: Path) -> None:
    paper = data["paper"]
    ours: Ours = data["ours"]
    figs = data["figs"]
    hyp = data["hyp"]
    frozen = data["frozen"]
    d = Doc()
    d.add_page()

    d.text_block("Отчёт о воспроизведении результатов статьи", 18, bold=True, gap=1)
    d.text_block("Rethinking the Value of Network Pruning (Liu, Sun, Zhou, Huang, Darrell; ICLR 2019, arXiv:1810.05270)", 11, gap=4)

    d.text_block("1. Цель и что воспроизводим", 13, bold=True)
    d.text_block(
        "Статья утверждает, что для структурированного прунинга дообучение унаследованных весов не лучше, "
        "чем обучение той же архитектуры с нуля при сопоставимом бюджете. Найденная прунингом архитектура, а не веса, "
        "даёт выигрыш. Для нерегулярного прунинга авторы видят исключение на ImageNet и при высокой разреженности. "
        "Я проверяю это на CIFAR и формулирую четыре гипотезы."
    )
    d.table(
        [
            ["№", "Гипотеза", "Критерий"],
            ["H1", "Для структурированного прунинга обучение с нуля не хуже fine-tune", "SB−FT ≥ −0.2 п.п. хотя бы в 4 из 5 конфигов и SE−FT ≥ −0.3 п.п. хотя бы в 3 из 5; опровержение — SB < FT − 0.5 в двух и более"],
            ["H1c", "Winning ticket не лучше случайной инициализации при lr 0.1", "|WT−SE| < 0.3 п.п. хотя бы в 2 из 3 конфигов L1"],
            ["H2", "Дольше обучение (Scratch-B) помогает; выигрыш растёт с долей прунинга", "SB−SE ≥ +0.1 в 70% пар с E_B > 160; SL-R56 SB−FT растёт с 40% до 80% и ≥ +1.0 на 80%"],
            ["H3", "Найденная архитектура лучше uniform на VGG, на ResNet близка к ней", "VGG: SL−UF ≥ +0.2 в 2 из 3; guided близок к SL; ResNet: |SL−UF| < 0.3 в 2 из 3"],
            ["H4", "Нерегулярный прунинг: Scratch-B не хуже при 80%, хуже при 95%", "UN-V19-80 SB−FT ≥ −0.3 и UN-R56-95 SB−FT ≤ −0.5"],
        ],
        [10, 60, 106],
        size=8,
        align="LEFT",
    )

    d.text_block("2. Экспериментальная установка", 13, bold=True)
    d.text_block(
        "Данные: CIFAR-10 (основная часть). Сети: VGG-16, VGG-19 с BatchNorm и одним линейным слоем, ResNet-56 с шорткатом типа A (zero-padding). "
        "Методы: L1-norm filter pruning, Network Slimming, нерегулярный magnitude pruning (Han), uniform-baseline с тем же числом параметров."
    )
    d.text_block(
        "Режимы одной конфигурации: база (160 эпох) → прунинг → Fine-tune (40 эпох, постоянный lr) | Scratch-E (160 эпох, He-инициализация) | "
        "Scratch-B. Для Scratch-B число эпох E_B = 4·round(160·κ/4), где κ = min(2, F_L/F_P) и F — число MAC большой и малой сети; "
        "расписание lr растягивается пропорционально (спады на 50% и 75%). Потолок κ = 2 оставлен как в статье для ImageNet; "
        "на CIFAR это ограничение я ввожу сам и называю Scratch-B (cap 2×)."
    )
    d.text_block(
        "Общий цикл: SGD с Нестеровым, momentum 0.9, weight decay 1e-4, batch 128, lr 0.1, аугментация — случайный кроп с отступом 4 и "
        "горизонтальный флип, независимо для каждой картинки. Обучение в bf16, оценка в FP32 без TF32. В отчёт идёт точность последней эпохи; "
        "тест читается один раз на прогон. Выбор по лучшей эпохе на тесте не делается."
    )
    d.text_block(
        f"Гиперпараметры заморожены на dev-сплите (45 000 / 5 000, стратифицированно) и зафиксированы в conf/frozen_hparams.json, "
        f"хэш {frozen.get('hash')}: precision={frozen.get('precision')}, lr_ft={frozen.get('lr_ft')}, "
        f"λ Slimming для VGG {frozen.get('slim_lambda_vgg')}, для ResNet {frozen.get('slim_lambda_resnet')}. "
        "После заморозки гиперпараметры не подбирались."
    )
    d.text_block(
        "Прунинг: L1 — доля каналов по слою; для ResNet режутся только внутренние каналы блока (conv1/bn1), поток остатка не трогается. "
        "Slimming: L1-штраф на γ BatchNorm добавляется к градиенту в fp32, глобальный top-k по |γ|. "
        "Нерегулярный: глобальный one-shot top-k по |w| только для весов свёрток; маска применяется к градиенту и к весам после шага. "
        "Uniform: одна доля на все слои, подобранная так, чтобы число параметров совпало с архитектурой Slimming сида 1; uniform обучается с нуля 160 эпох (по runs.csv), как Scratch-E."
    )
    d.text_block(
        "Железо: одна RTX 5070 Ti (sm_120), PyTorch 2.11 + CUDA 12.8, без torch.compile. Две обучающие программы одновременно "
        "на этой карте не давали выигрыша, поэтому очередь шла в один поток."
    )

    d.text_block("3. Проверка базовых сетей (dev-сплит)", 13, bold=True)
    rows = [["Прогон", "Сеть", "Данные", "val, %", "Статья, %", "Разница"]]
    for r in data["dev"]:
        rows.append([r["run_id"], r["model"], r["dataset"], f"{float(r['val_acc']):.2f}", r["paper"], f"{float(r['gap']):+.2f}"])
    d.table(rows, [52, 26, 24, 24, 24, 26], size=8)
    d.text_block(
        "На dev-сплите (45k обучающих картинок) базовые сети на 0.8–0.95 п.п. ниже статьи на CIFAR-10 и на 2.3 п.п. ниже на CIFAR-100 (VGG-19). "
        "Часть разрыва — меньший обучающий набор; на полном сплите (тест, last epoch) базы такие: VGG-16 92.84, VGG-19 92.84, ResNet-56 93.03 против 93.63, 93.53, 93.14 в статье. "
        "bf16 против FP32 на VGG-16 даёт разницу −0.14 п.п., это внутри порога 0.3, поэтому оставлен bf16. "
        "Причину отставания от статьи я не искал; сравнение методов идёт внутри одного моего цикла. "
        "Гипотеза, не проверялась: weight decay 1e-4 мало для VGG с BatchNorm; просадка базы только у VGG (−0.7…−0.8 п.п. на тесте), у ResNet-56 −0.1 п.п."
    )

    d.add_page()
    d.text_block("4. Результаты", 13, bold=True)
    d.text_block("4.1. L1-norm filter pruning (таблица 1 статьи)", 11, bold=True)
    l1 = _comparison_table(
        ours,
        paper["table1_l1_cifar10"],
        [("Моя VGG-16-A*", "L1-V16A", "V16"), ("Моя ResNet-56-A*", "L1-R56A", "R56"), ("Моя ResNet-56-B*", "L1-R56B", "R56")],
        lambda r: f"Статья {r['pruned']}",
        with_macs=True,
    )
    d.table(l1, [34, 22, 22, 22, 22, 16, 16, 20], size=7.5)
    d.text_block(
        "Разности для моих прогонов считаются по нескруглённым значениям, затем округляются; для статьи — по опубликованным значениям с двумя знаками." " Колонка «−MACs, %» — сокращение MAC относительно базы того же прогона по таблице приложения; для строк статьи значения нет в PDF (конфигурации заданы ссылкой на Li et al.).",
        8.5,
    )
    d.text_block(
        "Звёздочка: конфигурации *-A и *-B в PDF заданы ссылкой на Li et al., без слоёв и долей, поэтому я выбрал их сам. "
        "Допущение: VGG-16-A* режет 50% первого и слоёв 8–13; ResNet-56-A* режет 30% внутренних каналов всех блоков; ResNet-56-B* режет 30% в стадии 1 и 60% в стадиях 2–3. "
        "Поэтому абсолютные числа строк статьи и моих несопоставимы, сравнивать нужно разности SE−FT и SB−FT. "
        "Мои: n = 2 сида (± — std по двум сидам). Статьи: 5 сидов. VGG-16-A* восстановлен мной; он может совпадать с конфигурацией Li et al., но это не проверено (Li et al. нет в приложенном PDF). "
        "Сопоставимость с таблицей 1 по степени прунинга не проверена."
    )
    d.text_block(
        "VGG-16-A*: разности близки к статье (у статьи SE−FT +0.21, SB−FT +0.37; у меня "
        + _delta(ours, "L1-V16A", "SE", "FT") + " и " + _delta(ours, "L1-V16A", "SB", "FT") + "). "
        "ResNet-56-A*: у статьи SE−FT −0.01, SB−FT +0.12; у меня " + _delta(ours, "L1-R56A", "SE", "FT") + " и " + _delta(ours, "L1-R56A", "SB", "FT") + ". "
        "ResNet-56-B*: у статьи −0.13 и +0.38; у меня " + _delta(ours, "L1-R56B", "SE", "FT") + " и " + _delta(ours, "L1-R56B", "SB", "FT") + ". "
        "B* режется сильнее всего (" + _macs_cut(ours, "L1-R56B", "R56") + "% MACs против " + _macs_cut(ours, "L1-R56A", "R56") + "% у A*), и именно там SB−FT наибольшая; это согласуется с H2 (разрыв растёт с долей прунинга). Здесь же у B* больше эпох Scratch-B: E_B = 316 против 232 у A*. "
        "Итого Scratch-B не хуже fine-tune в двух конфигах из трёх с запасом, в третьем (A*) проигрыш внутри порога шума. Scratch-E соответствует статье только на VGG-16-A*."
    )
    d.picture(figs["ours_h1"], 150)

    d.text_block("4.2. Network Slimming (таблица 4 статьи)", 11, bold=True)
    sl_rows = [r for r in paper["table4_slimming_cifar"] if r["model"] == "VGG-19" and r["dataset"] == "CIFAR-10"]
    sl = _comparison_table(
        ours,
        sl_rows,
        [
            ("Моя VGG-19 70%", "SL-V19-70", "V19"),
            ("Моя ResNet-56 60%", "SL-R56-60", "R56"),
            ("Моя ResNet-56 40% (n=1)", "SL-R56-40", "R56"),
            ("Моя ResNet-56 80% (n=1)", "SL-R56-80", "R56"),
        ],
        lambda r: f"Статья {r['model']} {r['ratio']}",
    )
    d.table(sl, [38, 26, 26, 26, 26, 17, 17], size=7.5)
    d.text_block(FOOTNOTE, 8.5)
    d.text_block(
        "VGG-19 с 70% совпадает с условиями статьи (доля каналов считается по всей сети). Для ResNet-56 доля считается только по внутренним каналам блоков (допущение), "
        "в статье для ResNet-56 Slimming не оценивался, там PreResNet-164 и DenseNet-40, которых у меня нет. Строки ResNet-56 — мои собственные точки."
    )
    d.text_block(
        "VGG-19, 70%: у статьи SB−FT = +0.21, SE−FT = −0.30; у меня SB−FT = " + _delta(ours, "SL-V19-70", "SB", "FT") + ", SE−FT = " + _delta(ours, "SL-V19-70", "SE", "FT") + ". "
        "SB−FT того же знака, но больше; SE−FT у меня положительная, у статьи отрицательная. На ResNet-56 разрыв растёт с долей прунинга и на 80% достигает "
        + _delta(ours, "SL-R56-80", "SB", "FT")
        + " п.п. (n=1). Fine-tune на 80% теряет относительно базы больше 6 п.п."
    )
    p13, p14 = paper["table13_aggressive_slimming_cifar10_preresnet56_80"], paper["table14_aggressive_l1_cifar10_resnet56_90"]
    d.text_block(
        "Приложение C статьи поддерживает этот результат на агрессивном прунинге: в таблице 13 (PreResNet-56, Slimming 80%) fine-tune "
        f"{p13['ft']:.2f}, Scratch-E {p13['se']:.2f}, Scratch-B {p13['sb']:.2f}; в таблице 14 (ResNet-56, L1-norm 90%) {p14['ft']:.2f} / {p14['se']:.2f} / {p14['sb']:.2f}. "
        "Направление то же, что у моей SL-R56-80 (fine-tune " + _ms(ours, "SL-R56-80", "FT") + ", Scratch-E " + _ms(ours, "SL-R56-80", "SE") + ", Scratch-B " + _ms(ours, "SL-R56-80", "SB") + "); сравнивать абсолютные числа нельзя: другие сети и способ задания доли прунинга."
    )
    d.picture(figs["ours_slim_resnet"], 110)

    d.text_block("4.3. Нерегулярный magnitude pruning (таблица 6 статьи)", 11, bold=True)
    un = _comparison_table(
        ours,
        [{**r} for r in paper["table6_unstructured_cifar10_vgg19"] if r["ratio"] in ("80%", "95%")],
        [
            ("Моя VGG-19 80% (n=1)", "UN-V19-80", "V19"),
            ("Моя VGG-19 95% (n=1)", "UN-V19-95", "V19"),
            ("Моя ResNet-56 95% (n=1)", "UN-R56-95", "R56"),
        ],
        lambda r: f"Статья VGG-19 {r['ratio']}",
    )
    d.table(un, [38, 26, 26, 26, 26, 17, 17], size=7.5)
    d.text_block(FOOTNOTE, 8.5)
    d.text_block(
        "Для VGG-19 сохраняется картина статьи: Scratch-B не хуже fine-tune и на 95%. Для ResNet-56 на 95% (47 тыс. параметров) "
        "Scratch-B выше fine-tune на 1.7 п.п. В статье на PreResNet-110 и DenseNet-BC-100 при 95% fine-tune лучше (таблица 6), а на VGG-19 при 95% Scratch-B лучше, как и у меня. "
        "Предполагаю, что fine-tune за 40 эпох на очень разреженной сети с постоянным lr 0.001 недообучается; я это не проверял. "
        "Таблица 16 статьи (приложение D) показывает, что удлинение fine-tune до 80 и 160 эпох не помогает, но только для умеренного L1-прунинга, не для 95% нерегулярного."
    )
    d.picture(figs["ours_unstructured"], 105)

    d.add_page()
    d.text_block("4.4. Найденная архитектура против uniform (раздел 5 статьи)", 11, bold=True)
    uf_rows = [["Конфиг", "Парам. SL (сид 1)", "SL, сид 1", "SL, среднее по сидам", "Парам. UF", "UF (сид 1)", "SL−UF (сид 1)"]]
    for model, pts in (("V19", ("30", "50", "70")), ("R56", ("40", "60", "80"))):
        for pt in pts:
            sl_cfg, uf_cfg = f"SL-{model}-{pt}", f"UF-{model}-{pt}"
            a1, u1 = ours.acc_seed(sl_cfg, "SE", 1), ours.acc_seed(uf_cfg, "UF", 1)
            n_sl = _n(ours, sl_cfg, "SE")
            uf_rows.append([
                f"{'VGG-19' if model == 'V19' else 'ResNet-56'} {pt}%",
                str(ours.params(sl_cfg, "SE", 1)),
                f"{a1:.2f}" if a1 is not None else "нет",
                _ms(ours, sl_cfg, "SE") + (f" (n={n_sl})" if n_sl > 1 else " (n=1)"),
                str(ours.params(uf_cfg, "UF")),
                f"{u1:.2f}" if u1 is not None else "нет",
                f"{a1 - u1:+.2f}" if a1 is not None and u1 is not None else "нет",
            ])
    d.table(uf_rows, [26, 24, 18, 36, 24, 18, 20], size=7.5)
    d.text_block(FOOTNOTE, 8.5)
    mean70 = ours.mean("SL-V19-70", "SE") - ours.acc_seed("UF-V19-70", "UF", 1)
    mean60 = ours.mean("SL-R56-60", "SE") - ours.acc_seed("UF-R56-60", "UF", 1)
    d.text_block(
        "Во всех сравнениях SL−UF (таблица, §5 и слайды) используется сид 1: uniform подогнан под архитектуру Slimming сида 1 и обучался один раз. "
        f"Разность по средним двум сидам Slimming была бы {mean70:+.2f} (VGG-19 70%) и {mean60:+.2f} (ResNet-56 60%); так сравнивать некорректно, потому что у сида 2 архитектура Slimming другая "
        f"({ours.params('SL-V19-70', 'SE', 2)} параметров против {ours.params('SL-V19-70', 'SE', 1)} у VGG-19 70%; {ours.params('SL-R56-60', 'SE', 2)} против {ours.params('SL-R56-60', 'SE', 1)} у ResNet-56 60%). "
        "Число параметров uniform подгоняется под архитектуру Slimming сида 1 с допуском до 2% (фактически 0.06–1.8%). Для каждой точки по 1 сиду на uniform, поэтому разности без доверительного интервала."
    )
    vg = ", ".join(f"{r['sl_minus_uf']:+.2f}" for r in hyp.get("H3_vgg_rows", []))
    rn = ", ".join(f"{r['point']}%: {float(r['abs_sl_minus_uf']):.2f}" for r in hyp.get("H3_resnet_rows", []))
    d.text_block(
        f"VGG-19: Slimming выигрывает {vg} п.п. при 30/50/70% — аналог рис. 3 на VGG-19 (в статье рис. 3, как и таблицы 7 и 21, построены на VGG-16). "
        f"На ResNet-56 разрыв |SL−UF| = {rn} п.п. (сид 1). В статье Slimming на ResNet-56 не оценивался; паритет со uniform там виден на PreResNet-164 (рис. 5 слева, таблица 18, ширины почти равномерны по стадиям). "
        "Для нерегулярного прунинга паритет показан на PreResNet-110 и DenseNet-40 (рис. 5 в центре и справа, таблицы 19–20). "
        "У меня на ResNet-56 разрыв остаётся, потому что найденные ширины неравномерны по стадиям: размах 0.37–0.39 (см. ниже)."
    )
    d.picture(figs["ours_vgg_uf"], 105)
    d.text_block("Доли сохранённых каналов по стадиям:", 9.5, bold=True)
    keep_rows = [["Конфиг", "Доли по стадиям", "Комментарий"]]
    for row in data["ours"].keeps:
        if not (row["cfg"].startswith("SL-") or row["cfg"].startswith("L1-")):
            continue
        note = ""
        if "later_more_pruned" in row:
            note = "поздние стадии режутся сильнее" if row["later_more_pruned"] else "—"
        if "range" in row:
            note = f"размах по стадиям {float(row['range']):.2f}"
        keep_rows.append([row["cfg"], row["rates"], note])
    d.table(keep_rows, [38, 80, 58], size=8, align="LEFT")
    d.text_block(
        "VGG-19 с Slimming режет поздние стадии и почти не трогает ранние: так же, как в таблице 21 статьи. "
        "ResNet-56 с моим λ = 1e-5 получает стадийный размах 0.37–0.39 (а не «почти равномерный»), поэтому паритет с uniform не ожидается."
    )
    d.picture(figs["ours_stage"], 105)

    d.text_block("4.5. Латентность на GPU", 11, bold=True)
    lat = data["ours"].latency
    names = []
    for r in lat:
        if r["run_id"] not in names:
            names.append(r["run_id"])
    lat_rows = [["Модель (сид 1)", "bs 1 FP32, мс", "bs 1 FP16, мс", "bs 256 FP32, мс", "bs 256 FP16, мс"]]
    for n in names:
        def get(batch, dtype):
            for r in lat:
                if r["run_id"] == n and r["batch"] == batch and r["dtype"] == dtype:
                    return f"{float(r['median_ms']):.2f}"
            return "нет"
        lat_rows.append([n.replace(".SE.s1", " (Scratch-E)").replace(".BASE.s1", " (база)"), get("1", "fp32"), get("1", "fp16"), get("256", "fp32"), get("256", "fp16")])
    d.table(lat_rows, [64, 28, 28, 28, 28], size=8, align="LEFT")
    d.text_block(
        "Медиана медиан по 5 повторам, FP32 без TF32 и FP16, CUDA events. Нерегулярные маски не мерялись: плотное ядро от маски не ускоряется. "
        "В FP32 на batch 1 ResNet-56, вероятно, упирается в запуск ядер, а не в FLOPs: прунинг 80% даёт 6.25 мс против 7.43 мс у базы. На batch 256 разница заметна: 9.47 мс против 16.48 мс."
    )
    d.text_block(_lat_fp16(lat, ours))
    d.picture(figs["ours_latency"], 150)

    d.add_page()
    d.text_block("5. Проверка гипотез", 13, bold=True)
    h_rows = [["Гипотеза", "Вердикт", "Данные"]]
    sbft = ", ".join(f"{r['cfg']} {_signed(r['sb_minus_ft'])}" for r in hyp.get("H1_rows", []))
    h_rows.append(["H1", hyp.get("H1", "нет"), f"SB−FT: {sbft}"])
    h_rows.append(["H1c", hyp.get("H1c", "нет"), "WT-прогоны вырезаны бюджетом"])
    h_rows.append(["H2, SB>SE", hyp.get("H2_budget", "нет"), f"{hyp.get('H2_budget_hits')} из {hyp.get('H2_budget_total')} пар с E_B>160"])
    curve = ", ".join(f"{r['cfg'].replace('SL-R56-', '')}%: {_signed(r['sb_minus_ft'])}" for r in hyp.get("H2_curve_rows", []))
    h_rows.append(["H2, кривая", hyp.get("H2_curve", "нет"), f"SB−FT на SL-R56: {curve}. Прогон без потолка 2× не запускался"])
    vgg = ", ".join(f"{r['point']}%: {_signed(r['sl_minus_uf'])}" for r in hyp.get("H3_vgg_rows", []))
    h_rows.append(["H3, VGG", hyp.get("H3_vgg", "нет"), f"SL−UF (сид 1): {vgg}"])
    h_rows.append(["H3, guided", hyp.get("H3_gd", "нет"), "GD-прогоны вырезаны бюджетом"])
    res = ", ".join(f"{r['point']}%: {float(r['abs_sl_minus_uf']):.2f}" for r in hyp.get("H3_resnet_rows", []))
    h_rows.append(["H3, ResNet", hyp.get("H3_resnet", "нет"), f"|SL−UF| (сид 1): {res}"])
    h_rows.append([
        "H4",
        hyp.get("H4", "нет"),
        f"UN-V19-80 SB−FT {_signed(hyp.get('H4_un80_sb_minus_ft'))}; UN-R56-95 SB−FT {_signed(hyp.get('H4_un95_sb_minus_ft'))}",
    ])
    d.table(h_rows, [26, 30, 120], size=8.5, align="LEFT")
    d.text_block(
        "Вердикт «не проверено» значит, что прогоны для критерия не запускались. Это не подтверждение и не опровержение. "
        "Выводы по H1, H2 и H3 (VGG) опираются на 2 сида для основных конфигов и 1 сид для остальных; std между сидами 0.01–0.87 п.п., "
        "то есть сопоставим с критериями в 0.2–0.3 п.п. Поэтому «подтверждена» для H1 следует читать как «согласуется со статьёй», но не как статистически строгое доказательство."
    )

    d.text_block("Знак SB−FT и SE−FT по сидам (5 основных конфигов; для SL-V19-70 и SL-R56-60 архитектура сида 2 отличается от сида 1):", 9.5, bold=True)
    sign_rows = [["Конфиг", "Сид", "SB−FT", "знак", "SE−FT", "знак"]]
    pos_sb = pos_se = total = 0
    for cfg in ("L1-V16A", "L1-R56A", "L1-R56B", "SL-V19-70", "SL-R56-60"):
        for seed in (1, 2):
            ft, se_, sb_ = (ours.acc_seed(cfg, m, seed) for m in ("FT", "SE", "SB"))
            if None in (ft, se_, sb_):
                continue
            total += 1
            pos_sb += sb_ - ft > 0
            pos_se += se_ - ft > 0
            sign_rows.append([cfg + ("*" if cfg.startswith("L1") else ""), str(seed), f"{sb_ - ft:+.2f}", "+" if sb_ - ft > 0 else "−", f"{se_ - ft:+.2f}", "+" if se_ - ft > 0 else "−"])
    d.table(sign_rows, [34, 14, 24, 14, 24, 14], size=8)
    d.text_block(
        f"SB−FT положительна в {pos_sb} из {total} пар (конфиг, сид), SE−FT — в {pos_se} из {total}. Знак отдельной разности при n=2 ненадёжен, см. строки таблицы с разными знаками у разных сидов. Разности считаются по нескруглённым значениям.",
        9.5,
    )

    d.text_block("6. Отклонения от статьи и ограничения", 13, bold=True)
    for item in [
        "Охват: только CIFAR-10 и VGG-16/19, ResNet-56. Не воспроизводились ImageNet, ThiNet, Regression, SSS, SFP, PreResNet, DenseNet, детекция, CIFAR-100 и transferred guided.",
        "Сиды: 2 у основных конфигов (L1×3, SL-V19-70, SL-R56-60), 1 у остальных; в статье 5. Сид 3 и все прогоны, зависящие от него, вырезаны по бюджету GPU.",
        "Вырезано бюджетом: сиды 2–3 у вторичных конфигов, CIFAR-100, guided pruning, winning ticket, UN-R56-80, Scratch-B без потолка 2×. Порядок и правило резки зафиксированы заранее.",
        "Конфиги L1-V16A*, L1-R56A*, L1-R56B* восстановлены мной: числа не сопоставимы с таблицей 1 по абсолютной величине; сопоставимость по степени прунинга с Li et al. не проверена.",
        "Доля прунинга ResNet-56 считается только по внутренним каналам блоков, поэтому не сравнима с таблицами 4 и 18.",
        "Scratch-B ограничен 2× (в статье такой потолок описан только для ImageNet). Потолок дополнительно урезает бюджет там, где F_L/F_P > 2 (по MAC из приложения): " + _capped(data) + ". Неограниченные варианты не запускались.",
        "Протокол fine-tune для Slimming в PDF не описан; я применил тот же, что и для остальных: 40 эпох, постоянный lr 0.001, выбранный на dev-сплите.",
        "Обучение в bf16 с автокастом; оценка в FP32. На dev разница bf16/FP32 −0.14 п.п.",
        "Базовые VGG на тесте ниже статьи на 0.7–0.8 п.п., ResNet-56 на 0.1 п.п. Причину не выяснял.",
        "Бюджет Scratch-B равен обучению большой сети без fine-tune, то есть меньше полного конвейера (обучение + fine-tune); авторы называют это уступкой в пользу прунинга. Стоимость fine-tune в сравнение не входит, как и в статье. "
        "Возражение «fine-tune 40 эпох мало» снимается приложением D статьи: таблица 16 (L1-norm, CIFAR-10), fine-tune 80 и 160 эпох заметного прироста не даёт (VGG-16-A: " + f"{paper['table16_ft_epochs_vgg16a']['ft40']:.2f} / {paper['table16_ft_epochs_vgg16a']['ft80']:.2f} / {paper['table16_ft_epochs_vgg16a']['ft160']:.2f}" + "). У меня fine-tune 40 эпох.",
    ]:
        d.text_block("•  " + item, 9.5, gap=1)

    d.text_block("7. Выводы", 13, bold=True)
    d.text_block(
        "Главное утверждение статьи для структурированного прунинга воспроизводится на CIFAR-10: во всех пяти основных конфигах Scratch-B не уступает fine-tune больше чем на 0.2 п.п., "
        "а в четырёх из пяти превосходит его; на ResNet-56 разрыв растёт с долей прунинга. Архитектура, найденная Slimming на VGG-19, даёт больше точности на параметр, чем uniform. "
        "Паритет Slimming с uniform на ResNet-56 не воспроизвёлся (в статье паритет показан на PreResNet-164, а не на ResNet-56). "
        "Исключение для нерегулярного прунинга в статье — ImageNet и 95% на PreResNet-110 и DenseNet-BC-100; их я не проверял. На VGG-19 при 95% и у статьи, и у меня Scratch-B ≥ fine-tune. "
        "H4 опиралась на перенос поведения PreResNet-110 на ResNet-56; это допущение, а не результат статьи, и в таком виде H4 опровергнута (SB−FT " + f"{hyp.get('H4_un95_sb_minus_ft', 0):+.2f}" + " п.п. при 95%). "
        "Расхождения предполагаю объяснять различием архитектур и недообучением fine-tune; это не проверялось. Winning ticket и guided pruning не проверены."
    )

    d.text_block("8. Воспроизводимость", 13, bold=True)
    d.text_block(
        f"Гиперпараметры зафиксированы тегом git hparams-v1 (хэш {frozen.get('hash')}). Все числа таблиц сгенерированы из report/tables/*.csv|json (src.aggregate, scripts/latency.py); "
        "числа статьи лежат в report/paper_tables.json и перенесены из PDF. Инварианты агрегатора ("
        + ("все прошли" if data["inv"].get("ok") else "есть нарушения")
        + "): одинаковый хэш гиперпараметров во всех прогонах, число эпох и вехи lr по плану, одна оценка на тесте, E_B в Scratch-B равно рассчитанному, разные init-отпечатки у разных сидов и у Scratch-E против fine-tune, "
        "нет полей best_*. Тесты в tests/ проходят полностью без пропусков. Команды запуска см. в README.MD."
    )
    last = data["plans"].get("plan_f2") or data["plans"].get("plan_f1") or {}
    if last:
        runs = last.get("runs", [])
        d.text_block(
            f"Бюджет GPU: допуск {float(last.get('allowance_h', 0)):.1f} ч, включено прогонов {sum(1 for r in runs if r.get('enabled'))} из {len(runs)}, "
            f"вырезано {sum(1 for r in runs if not r.get('enabled'))}.",
            9.5,
        )

    d.add_page()
    d.text_block("Приложение. Все результаты на тесте (последняя эпоха, mean ± std по сидам)", 12, bold=True)
    d.text_block(
        "BASE — база, обученная обычным образом. BASES — база, обученная с L1-штрафом Slimming на γ BatchNorm; из неё берутся архитектуры Slimming и она служит «исходной» сетью в строках SL. "
        "PRUNE — шаг прунинга без обучения (точности нет). FT, SE, SB, UF — fine-tune, Scratch-E, Scratch-B, uniform с нуля. MAC — умножения-сложения на одну картинку 32×32.",
        9,
    )
    rows = [["Конфиг", "Режим", "n", "Точность, %", "Параметров", "MAC"]]
    for r in data["summary"]:
        rows.append([r["cfg"], r["mode"], r["n"], r["test_acc"], r["params"], r["macs"]])
    d.table(rows, [38, 20, 10, 40, 34, 34], size=7.5)
    d.output(str(path))


def main() -> None:
    if not FONT.exists() or not FONT_B.exists():
        raise SystemExit(f"DejaVu not found at {FONT}")
    data = _load()
    out = ROOT / "report"
    out.mkdir(parents=True, exist_ok=True)
    write_slides(data, out / "slides.pdf")
    write_report(data, out / "report.pdf")
    print(f"wrote {out / 'slides.pdf'} and {out / 'report.pdf'}")


if __name__ == "__main__":
    main()
