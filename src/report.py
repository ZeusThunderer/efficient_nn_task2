"""Russian report and slides. Every number is read from report/tables or dev/summary.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fpdf import FPDF

from src.aggregate import build
from src.paths import FROZEN_HPARAMS, ROOT

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_B = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
TABLES = ROOT / "report" / "tables"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load() -> dict:
    build(TABLES)
    hyp_path = TABLES / "hypotheses.json"
    hyp = json.loads(hyp_path.read_text(encoding="utf-8")) if hyp_path.exists() else {}
    inv_path = TABLES / "invariants.json"
    inv = json.loads(inv_path.read_text(encoding="utf-8")) if inv_path.exists() else {"ok": False, "errors": []}
    keeps_path = TABLES / "stage_keep.json"
    keeps = json.loads(keeps_path.read_text(encoding="utf-8")) if keeps_path.exists() else []
    frozen = {}
    if FROZEN_HPARAMS.exists():
        frozen = json.loads(FROZEN_HPARAMS.read_text(encoding="utf-8"))
    plans = {}
    for name in ("f0", "plan_f0", "plan_f1", "plan_f2"):
        path = ROOT / "budget" / f"{name}.json"
        if path.exists():
            plans[name] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "summary": _read_csv(TABLES / "summary.csv"),
        "runs": _read_csv(TABLES / "runs.csv"),
        "dev": _read_csv(ROOT / "dev" / "summary.csv"),
        "latency": _read_csv(TABLES / "latency.csv"),
        "hyp": hyp,
        "inv": inv,
        "keeps": keeps,
        "frozen": frozen,
        "plans": plans,
    }


def _num(value) -> str:
    if value is None or value == "":
        return "нет прогона"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


class Doc(FPDF):
    def __init__(self, landscape: bool = False):
        super().__init__(orientation="L" if landscape else "P", unit="mm", format="A4")
        self.add_font("DejaVu", "", str(FONT))
        self.add_font("DejaVu", "B", str(FONT_B))
        self.set_auto_page_break(auto=True, margin=15)

    def h1(self, text: str) -> None:
        self.set_font("DejaVu", "B", 16)
        self.multi_cell(0, 8, text)
        self.ln(2)

    def h2(self, text: str) -> None:
        self.set_font("DejaVu", "B", 13)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def p(self, text: str) -> None:
        self.set_font("DejaVu", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def small(self, text: str) -> None:
        self.set_font("DejaVu", "", 9)
        self.multi_cell(0, 5, text)
        self.ln(1)


def _lookup(summary: list[dict], cfg: str, mode: str) -> str:
    for row in summary:
        if row["cfg"] == cfg and row["mode"] == mode:
            return row["test_acc"] or "нет прогона"
    return "нет прогона"


def write_report(data: dict, path: Path) -> None:
    pdf = Doc()
    pdf.add_page()
    pdf.h1("Воспроизведение: Rethinking the Value of Network Pruning")
    pdf.p("Liu, Sun, Zhou, Huang, Darrell. ICLR 2019, arXiv:1810.05270. Отчёт собран из CSV, числа не переписывались вручную.")
    pdf.h2("Тезис")
    pdf.p(
        "Для структурированного прунинга дообучение унаследованных весов не лучше, чем обучение "
        "той же архитектуры с нуля при сопоставимом бюджете. Ценность прунинга — в архитектуре. "
        "Нерегулярный magnitude-прунинг на высокой разреженности — исключение. ImageNet в эту работу не входит."
    )
    pdf.h2("Протокол")
    pdf.p(
        "Один цикл обучения. База 160 эпох, SGD Nesterov, momentum 0.9, weight decay 1e-4, batch 128, lr 0.1, "
        "спады в 50% и 75% эпох. Fine-tune: 40 эпох, постоянный lr. Scratch-E: 160 эпох, инициализация He. "
        "Scratch-B (cap 2×): E_B = 4 * round(160 * min(2, F_L/F_P) / 4). Стоимость fine-tune в сравнение бюджетов не входит. "
        "FLOPs в таблицах — это MAC, без BN, ReLU, пулинга и сложений. Отчётная точность — последняя эпоха, не лучший чекпоинт."
    )
    frozen = data["frozen"]
    pdf.h2("Замороженные гиперпараметры")
    if frozen:
        pdf.p(
            f"precision={frozen.get('precision')}, lr_ft={frozen.get('lr_ft')}, "
            f"lambda VGG={frozen.get('slim_lambda_vgg')}, lambda ResNet={frozen.get('slim_lambda_resnet')}, "
            f"hash={frozen.get('hash')}."
        )
    else:
        pdf.p("Файл conf/frozen_hparams.json отсутствует: волна на полном тесте не запускалась.")
    pdf.h2("Отклонения от статьи")
    pdf.p(
        "Конфиги L1-V16A*, L1-R56A*, L1-R56B* помечены звёздочкой: точные слои и доли в PDF не заданы. "
        "Доля каналов ResNet считается только по внутренним каналам блоков и не сопоставима с Table 4. "
        "Scratch-B ограничен двумя разами; для SL-R56-80 есть отдельный прогон без потолка (SBU). "
        "Рисунок 3 построен на VGG-19, guided — только на CIFAR-10. Три сида вместо пяти. "
        "Протокол fine-tune для Slimming в PDF не описан; здесь он тот же, что и для остальных методов."
    )
    pdf.h2("Dev-split, не тест")
    if data["dev"]:
        for row in data["dev"]:
            pdf.small(
                f"{row['run_id']}: val {row['val_acc']}  статья {row['paper']}  gap {row['gap']}"
            )
    else:
        pdf.p("dev/summary.csv нет.")

    pdf.add_page()
    pdf.h2("Тест, mean±std по сидам")
    if not data["summary"]:
        pdf.p("Полных прогонов нет.")
    else:
        for row in data["summary"]:
            pdf.small(
                f"{row['cfg']} {row['mode']} {row['dataset']}: {row['test_acc']}  n={row['n']}  "
                f"params={row['params']}  macs={row['macs']}"
            )

    hyp = data["hyp"]
    pdf.add_page()
    pdf.h2("Гипотезы")
    pdf.p(f"H1 (Scratch-B и Scratch-E против fine-tune): {hyp.get('H1', 'нет данных')}.")
    for row in hyp.get("H1_rows", []):
        pdf.small(f"{row['cfg']}: SB−FT {_num(row['sb_minus_ft'])} п.п., SE−FT {_num(row['se_minus_ft'])} п.п.")
    pdf.p(f"H1c (winning ticket против Scratch-E при lr 0.1): {hyp.get('H1c', 'нет данных')}.")
    for row in hyp.get("H1c_rows", []):
        pdf.small(f"{row['cfg']}: WT−SE {_num(row['wt_minus_se'])} п.п.")
    pdf.p(
        f"H2, доля пар с E_B>160, где SB−SE ≥ +0.1 п.п.: {hyp.get('H2_budget', 'нет данных')} "
        f"({hyp.get('H2_budget_hits', 0)}/{hyp.get('H2_budget_total', 0)})."
    )
    pdf.p(f"H2, кривая SL-R56 40/60/80, Scratch-B cap 2×: {hyp.get('H2_curve', 'нет данных')}.")
    for row in hyp.get("H2_curve_rows", []):
        pdf.small(f"{row['cfg']}: SB−FT {_num(row['sb_minus_ft'])} п.п.")
    pdf.p(f"SBU−FT на SL-R56-80 (без потолка, контроль): {_num(hyp.get('H2_sbu_minus_ft'))} п.п.")
    pdf.p(f"H3, VGG Slimming против uniform: {hyp.get('H3_vgg', 'нет данных')}.")
    for row in hyp.get("H3_vgg_rows", []):
        pdf.small(f"VGG {row['point']}%: SL−UF {_num(row['sl_minus_uf'])} п.п.")
    pdf.p(f"H3, guided: {hyp.get('H3_gd', 'нет данных')}.")
    for row in hyp.get("H3_gd_rows", []):
        pdf.small(f"GD {row['point']}%: GD−UF {_num(row['gd_minus_uf'])}, GD−SL {_num(row['gd_minus_sl'])}")
    pdf.p(f"H3, ResNet |SL−UF|: {hyp.get('H3_resnet', 'нет данных')}.")
    for row in hyp.get("H3_resnet_rows", []):
        pdf.small(f"ResNet {row['point']}%: |SL−UF| {_num(row['abs_sl_minus_uf'])} п.п.")
    pdf.p(
        f"H4: {hyp.get('H4', 'нет данных')}. UN-V19-80 SB−FT {_num(hyp.get('H4_un80_sb_minus_ft'))} п.п., "
        f"UN-R56-95 SB−FT {_num(hyp.get('H4_un95_sb_minus_ft'))} п.п."
    )

    pdf.add_page()
    pdf.h2("Доли сохранённых каналов, сид 1, Scratch-E")
    if not data["keeps"]:
        pdf.p("Нет архитектур.")
    for row in data["keeps"]:
        extra = ""
        if "later_more_pruned" in row:
            extra = f" поздние стадии сильнее: {row['later_more_pruned']}"
        if "range" in row:
            extra = f" размах стадий {float(row['range']):.3f}"
        pdf.small(f"{row['cfg']}: {row['rates']}{extra}")
    pdf.h2("Латентность GPU")
    pdf.p("Замеры после обучения, FP32 с выключенным TF32 и FP16, batch 1 и 256. Нерегулярные маски не мерялись: плотное ядро от маски не ускоряется.")
    if not data["latency"]:
        pdf.p("latency.csv нет.")
    for row in data["latency"]:
        pdf.small(
            f"{row.get('run_id')} bs={row.get('batch')} {row.get('dtype')}: "
            f"median {row.get('median_ms')} ms, IQR {row.get('iqr_ms')} ms"
        )
    pdf.h2("Бюджет")
    f0 = data["plans"].get("f0", {})
    if f0:
        pdf.p(f"G1: t_ep={f0.get('t_ep')}, phi={f0.get('phi')}, K={f0.get('k')}, peak_mb={f0.get('peak_mb')}.")
    for name in ("plan_f0", "plan_f1", "plan_f2"):
        plan = data["plans"].get(name)
        if not plan:
            continue
        enabled = sum(1 for row in plan.get("runs", []) if row.get("enabled"))
        pdf.small(f"{name}: allowance {plan.get('allowance_h')} ч, план {plan.get('cost_h')} ч, включено {enabled}.")
    pdf.h2("Инварианты")
    if data["inv"].get("ok"):
        pdf.p("Проверки агрегатора прошли.")
    else:
        pdf.p("Агрегатор нашёл нарушения:")
        for err in data["inv"].get("errors", []):
            pdf.small(err)
    pdf.output(str(path))


def write_slides(data: dict, path: Path) -> None:
    pdf = Doc(landscape=True)
    hyp = data["hyp"]
    slides = [
        (
            "Rethinking the Value of Network Pruning",
            "Воспроизведение arXiv:1810.05270 на CIFAR-10/100.\n"
            "Сравнение fine-tune, Scratch-E и Scratch-B (cap 2×), а не абсолютная точность.",
        ),
        (
            "Тезис",
            "Структурированный прунинг ценен архитектурой, а не унаследованными весами.\n"
            "При честном бюджете обучение с нуля не хуже дообучения.\n"
            "Нерегулярный прунинг на 95% — ожидаемое исключение.",
        ),
        (
            "Что сравниваем",
            "BASE 160 эпох → прунинг каналов или маска → FT 40 эпох.\n"
            "Scratch-E: та же архитектура, 160 эпох, He init.\n"
            "Scratch-B (cap 2×): эпохи подогнаны под FLOPs исходной сети, потолок κ=2.\n"
            "Uniform и guided — контрольные архитектуры с тем же числом параметров.",
        ),
        (
            "Честность",
            "Один train loop. Отличаются инициализация, длина и расписание lr.\n"
            "Точность — последняя эпоха. Тест читается один раз.\n"
            "Гиперпараметры заморожены до волны и не подкручиваются по тесту.\n"
            f"hash={data['frozen'].get('hash', 'нет')}, precision={data['frozen'].get('precision', 'нет')}, lr_ft={data['frozen'].get('lr_ft', 'нет')}.",
        ),
        (
            "Где мы отошли от PDF",
            "L1-V16A*, L1-R56A*, L1-R56B*: слои не заданы в статье.\n"
            "Доля ResNet — только внутренние каналы, это не Table 4.\n"
            "Scratch-B с потолком 2×; SBU — контроль без потолка.\n"
            "VGG-19 вместо VGG-16 для кривой Slimming. n=3, не 5. ImageNet нет.",
        ),
        (
            "Sanity на val, не на тесте",
            "\n".join(
                f"{row['run_id']}: {row['val_acc']} (статья {row['paper']}, gap {row['gap']})"
                for row in data["dev"]
            )
            or "Dev-прогонов нет.",
        ),
        (
            "H1. Пять основных конфигов",
            "\n".join(
                f"{row['cfg']}: SB−FT {_num(row['sb_minus_ft'])}, SE−FT {_num(row['se_minus_ft'])}"
                for row in hyp.get("H1_rows", [])
            )
            or "Нет пар.",
        ),
        (
            "H1. Вердикт",
            f"H1: {hyp.get('H1', 'нет данных')}.\n"
            "Правило: SB−FT ≥ −0.2 п.п. хотя бы в 4 из 5 и SE−FT ≥ −0.3 хотя бы в 3 из 5.\n"
            "Опровержение: SB ниже FT на 0.5 п.п. хотя бы в двух конфигах.\n"
            f"H1c, |WT−SE| < 0.3 на L1 при lr 0.1: {hyp.get('H1c', 'нет данных')}.",
        ),
        (
            "H2. Бюджет и кривая Slimming",
            f"Доля SB−SE ≥ +0.1 при E_B>160: {hyp.get('H2_budget', 'нет данных')} "
            f"({hyp.get('H2_budget_hits', 0)}/{hyp.get('H2_budget_total', 0)}).\n"
            f"SL-R56 40→60→80, cap 2×: {hyp.get('H2_curve', 'нет данных')}.\n"
            + "\n".join(f"{row['cfg']}: SB−FT {_num(row['sb_minus_ft'])}" for row in hyp.get("H2_curve_rows", []))
            + f"\nSBU−FT: {_num(hyp.get('H2_sbu_minus_ft'))}.",
        ),
        (
            "H3. Архитектура, не процедура",
            f"VGG SL−UF: {hyp.get('H3_vgg', 'нет данных')}.\n"
            + "\n".join(f"{row['point']}%: {_num(row['sl_minus_uf'])}" for row in hyp.get("H3_vgg_rows", []))
            + f"\nGuided: {hyp.get('H3_gd', 'нет данных')}. ResNet: {hyp.get('H3_resnet', 'нет данных')}.",
        ),
        (
            "H4. Нерегулярный прунинг",
            f"Вердикт: {hyp.get('H4', 'нет данных')}.\n"
            f"UN-V19-80 SB−FT: {_num(hyp.get('H4_un80_sb_minus_ft'))} (ожидаем ≥ −0.3).\n"
            f"UN-R56-95 SB−FT: {_num(hyp.get('H4_un95_sb_minus_ft'))} (ожидаем ≤ −0.5).\n"
            "Стена времени у маски не падает: ядро остаётся плотным.",
        ),
        (
            "Где режутся каналы",
            "\n".join(
                f"{row['cfg']}: {row['rates']}"
                for row in data["keeps"]
            )
            or "Нет данных. Ожидание: у VGG поздние стадии режутся сильнее, у ResNet стадийный размах ≤ 0.15.",
        ),
        (
            "Латентность",
            "\n".join(
                f"{row['run_id']} bs {row['batch']} {row['dtype']}: {row['median_ms']} ms"
                for row in data["latency"][:12]
            )
            or "Замер после всех тренировок. На batch 1 ResNet упирается в запуск ядра, не во FLOPs.",
        ),
        (
            "Бюджет GPU",
            (
                f"Smoke phi={data['plans'].get('f0', {}).get('phi', 'нет')}, K={data['plans'].get('f0', {}).get('k', 'нет')}.\n"
                "Сначала режутся хвостовые классы: лишние сиды P2a, CIFAR-100, guided, контроль 80%, SBU, winning ticket.\n"
                "160/40 эпох, κ=2 и оценка последней эпохи не режутся."
            ),
        ),
        (
            "Вывод",
            "Смотрите вердикты H1–H4 на предыдущих слайдах: они посчитаны из test_acc последней эпохи.\n"
            "Если абсолютная точность ниже статьи, сравнение методов всё равно действительно внутри одного цикла.\n"
            "Вырезанный подкритерий — это «не проверено», а не подтверждение.",
        ),
    ]
    for title, body in slides:
        pdf.add_page()
        pdf.h1(title)
        pdf.p(body)
    pdf.output(str(path))


def main() -> None:
    if not FONT.exists() or not FONT_B.exists():
        raise SystemExit(f"DejaVu not found at {FONT}")
    data = _load()
    report = ROOT / "report" / "report.pdf"
    slides = ROOT / "report" / "slides.pdf"
    report.parent.mkdir(parents=True, exist_ok=True)
    write_report(data, report)
    write_slides(data, slides)
    print(f"wrote {report} and {slides}")


if __name__ == "__main__":
    main()
