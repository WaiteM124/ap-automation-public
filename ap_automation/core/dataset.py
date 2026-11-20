"""
Synthetic dataset generator.

Produces invoices in three tiers plus labelled governance anomalies:

  routine    known vendors, descriptive text, entity and cost centre consistent with
             the chart of accounts and the vendor's permitted entities
  near_edge  (a) known vendors with VAGUE descriptions - the text carries no signal and
                 some of these are coded to the vendor's secondary GL, so the gate is
                 genuinely tested
             (b) vendors that are in the vendor master but have NO history yet
  hard_edge  vendors that are not in the vendor master (new names, foreign currency)

A small share of routine invoices carries an injected `anomaly` that a specialist agent
must catch (wrong entity, wrong cost centre, blocked VAT, reverse charge). Every row has
a ground-truth GL and, if applicable, an anomaly label, so the evaluation can measure
accuracy, false positives and governance catch-rate instead of asserting them.

Earlier versions drew entity and cost centre at random, which made the specialist-agent
checks meaningless noise, and gave every vendor a single GL, which made ~100% accuracy
nearly guaranteed. Neither is true any more.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from ap_automation.core.chart_of_accounts import CHART, GL_NAMES
from ap_automation.core.models import InvoiceCategory, InvoiceLine

# Re-exported for the UI / older callers
GL_CODES = GL_NAMES
COST_CENTRES = ["CC100", "CC200", "CC300", "CC400", "CC500"]
ENTITIES = ["UK001", "UK002", "IE001"]


@dataclass(frozen=True)
class VendorSpec:
    canonical_id: str
    raw_names: tuple[str, ...]
    gl: str
    amount_range: tuple[float, float]
    entities: tuple[str, ...] = ("UK001",)
    secondary_gl: str | None = None
    secondary_share: float = 0.0


# Vendors with history in the seeded corpus
KNOWN_VENDORS: list[VendorSpec] = [
    VendorSpec("MERIDIAN_FACILITIES", ("Meridian Facilities Ltd", "Meridian Facilities", "Meridian Fac. Ltd"),
               "6300", (200, 5000), ("UK001", "UK002")),
    VendorSpec("TECHSOURCE_IT", ("TechSource IT Solutions", "TechSource IT", "Tech Source Ltd"),
               "6200", (500, 15000), ("UK001", "UK002"), secondary_gl="7100", secondary_share=0.2),
    VendorSpec("GLOBAL_OFFICE", ("Global Office Supplies", "Global Office Ltd", "Global Office"),
               "6100", (50, 800), ("UK001",)),
    VendorSpec("HORIZON_CONSULT", ("Horizon Consulting Group", "Horizon Consulting", "Horizon Group Ltd"),
               "6400", (2000, 20000), ("UK001",), secondary_gl="6800", secondary_share=0.25),
    VendorSpec("SWIFT_TRAVEL", ("Swift Travel Services", "Swift Travel Ltd", "SwiftTravel"),
               "6500", (150, 3000), ("UK001", "UK002")),
    VendorSpec("POWERLINE_UTILS", ("Powerline Utilities", "Powerline Util. Ltd", "Powerline"),
               "6600", (300, 2000), ("UK001",)),
    VendorSpec("BRANDWORKS_MKT", ("Brandworks Marketing", "BrandWorks Ltd", "Brandworks Agency"),
               "6700", (1000, 20000), ("UK001",)),
    VendorSpec("LEARN_FORWARD", ("Learn Forward Training", "LearnForward Ltd", "Learn Forward"),
               "6800", (500, 5000), ("UK002",)),
    VendorSpec("APEX_EQUIPMENT", ("Apex Equipment Co", "Apex Equipment Ltd", "Apex Co."),
               "7100", (5000, 80000), ("UK001",)),
]

# In the vendor master but with NO history: must be reviewed until a human has coded a few
NEW_TO_CORPUS_VENDORS: list[VendorSpec] = [
    VendorSpec("NOVACORP_SVCS", ("NovaCorp Services Ltd",), "6400", (1000, 10000)),
    VendorSpec("DATABRIDGE_IT", ("DataBridge IT Solutions",), "6200", (2000, 20000)),
    VendorSpec("CLEANZONE_FAC", ("CleanZone Facilities",), "6300", (300, 3000)),
    VendorSpec("SUMMIT_TRAIN", ("Summit Training Academy",), "6800", (800, 6000)),
    VendorSpec("GREENPATH_UTL", ("GreenPath Utilities",), "6600", (400, 2500)),
]

# NOT in the vendor master
UNKNOWN_VENDORS: list[tuple[str, str, tuple[float, float]]] = [
    ("Zephyr Global Holdings", "6400", (5000, 50000)),
    ("Arcturus Supply Co", "6100", (100, 500)),
    ("Pulsar Tech GmbH", "6200", (3000, 30000)),
    ("Meridian East Europe s.r.o.", "6300", (200, 2000)),
    ("Solaris Energy Partners", "6600", (500, 5000)),
    ("Apex Dental Supplies", "6100", (80, 900)),
]

DESCRIPTIONS: dict[str, list[str]] = {
    "6100": ["Office stationery and printer consumables", "Paper, pens and office supplies Q{q}",
             "Monthly office supplies order", "Printer cartridges and stationery",
             "Office consumables replenishment"],
    "6200": ["Annual software licence renewal", "Cloud infrastructure services Q{q}",
             "IT support and maintenance contract", "Software subscription renewal",
             "Cybersecurity platform licence", "SaaS platform annual fee"],
    "6300": ["Office cleaning services monthly", "Building maintenance and repairs",
             "HVAC servicing and inspection", "Facility management services Q{q}",
             "Pest control and grounds maintenance"],
    "6400": ["Management consulting services", "Legal advisory fees Q{q}",
             "Audit and assurance services", "HR consulting project", "Financial advisory retainer"],
    "6500": ["Business travel accommodation", "Flight and hotel Q{q}", "Rail travel and expenses",
             "Client meeting travel costs", "Conference accommodation and travel"],
    "6600": ["Electricity supply monthly", "Gas and electricity Q{q}", "Water rates quarterly",
             "Utilities combined bill", "Energy supply invoice"],
    "6700": ["Digital marketing campaign", "Social media advertising Q{q}",
             "Brand design and production", "PR retainer monthly", "Marketing materials and print"],
    "6800": ["Staff training programme", "Leadership development course", "Compliance training Q{q}",
             "Technical skills workshop", "Professional development programme"],
    "7100": ["Server hardware purchase", "Network equipment installation",
             "Production machinery purchase", "Capital equipment acquisition",
             "Laptop and workstation hardware"],
    "7200": ["Office refurbishment and fit-out", "Leasehold improvement programme"],
}

# Text that carries no coding signal
VAGUE_DESCRIPTIONS = [
    "Services as per agreement", "Various items - see attached", "Invoice for services rendered",
    "Monthly charge", "As discussed", "Project work",
]

# Injected governance anomalies: label -> description override / field override
ANOMALY_TYPES = ("wrong_entity", "wrong_cost_centre", "blocked_vat", "reverse_charge")
ANOMALY_RATE = 0.06


def _description(rng: random.Random, gl: str) -> str:
    return rng.choice(DESCRIPTIONS[gl]).format(q=rng.randint(1, 4))


def _pick_gl(rng: random.Random, v: VendorSpec) -> str:
    if v.secondary_gl and rng.random() < v.secondary_share:
        return v.secondary_gl
    return v.gl


def _line(rng, *, vendor_name, gl, amount_range, entity, currency="GBP", description=None,
          category, cost_centre=None, anomaly=None) -> InvoiceLine:
    return InvoiceLine(
        invoice_id=UUID(int=rng.getrandbits(128), version=4),
        raw_vendor_name=vendor_name,
        description=description or _description(rng, gl),
        amount=round(rng.uniform(*amount_range), 2),
        currency=currency,
        entity_id=entity,
        cost_centre=cost_centre or CHART[gl].default_cost_centre,
        category=category,
        ground_truth_gl=gl,
        anomaly=anomaly,
    )


def _routine(rng: random.Random, vendors: list[VendorSpec]) -> InvoiceLine:
    v = rng.choice(vendors)
    gl = _pick_gl(rng, v)
    inv = _line(rng, vendor_name=rng.choice(v.raw_names), gl=gl, amount_range=v.amount_range,
                entity=rng.choice(v.entities), category=InvoiceCategory.ROUTINE)
    if rng.random() >= ANOMALY_RATE:
        return inv

    kind = rng.choice(ANOMALY_TYPES)
    if kind == "wrong_entity":
        inv.entity_id = "IE001"  # caught by the entity agent, or by the VAT agent (non-UK entity)
    elif kind == "wrong_cost_centre":
        inv.cost_centre = rng.choice([c for c in COST_CENTRES if c != CHART[gl].default_cost_centre])
    elif kind == "blocked_vat":
        inv.description = "Client hospitality dinner and entertainment"
    elif kind == "reverse_charge":
        inv.description = "Construction works - scaffolding and groundworks"
    inv.anomaly = kind
    return inv


def generate_invoices(
    n_routine: int = 600,
    n_near_edge: int = 150,
    n_hard_edge: int = 50,
    seed: int = 42,
) -> tuple[list[InvoiceLine], list[dict]]:
    """Return (invoices in randomised stream order, vendor-master rows)."""
    rng = random.Random(seed)
    invoices: list[InvoiceLine] = []

    for _ in range(n_routine):
        invoices.append(_routine(rng, KNOWN_VENDORS))

    for _ in range(n_near_edge):
        if rng.random() < 0.5:  # known vendor, vague text, sometimes the secondary GL
            v = rng.choice(KNOWN_VENDORS)
            gl = _pick_gl(rng, v)
            invoices.append(_line(
                rng, vendor_name=rng.choice(v.raw_names), gl=gl, amount_range=v.amount_range,
                entity=rng.choice(v.entities), category=InvoiceCategory.NEAR_EDGE,
                description=rng.choice(VAGUE_DESCRIPTIONS),
            ))
        else:  # vendor in the master, no history yet
            v = rng.choice(NEW_TO_CORPUS_VENDORS)
            invoices.append(_line(
                rng, vendor_name=v.raw_names[0], gl=v.gl, amount_range=v.amount_range,
                entity=v.entities[0], category=InvoiceCategory.NEAR_EDGE,
            ))

    for i in range(n_hard_edge):
        name, gl, amount_range = UNKNOWN_VENDORS[i % len(UNKNOWN_VENDORS)]
        invoices.append(_line(
            rng, vendor_name=name, gl=gl, amount_range=amount_range,
            entity=rng.choice(ENTITIES[:2]), currency=rng.choice(["GBP", "EUR", "USD", "CHF"]),
            category=InvoiceCategory.HARD_EDGE,
        ))

    rng.shuffle(invoices)

    vendor_rows = [
        {"raw_name": raw, "canonical_id": v.canonical_id}
        for v in KNOWN_VENDORS + NEW_TO_CORPUS_VENDORS
        for raw in v.raw_names
    ]
    return invoices, vendor_rows


def generate_history(per_vendor: int = 15, seed: int = 7) -> list[InvoiceLine]:
    """Historical, human-coded lines for the known vendors (the seed corpus). Anomaly-free."""
    rng = random.Random(seed)
    history: list[InvoiceLine] = []
    for v in KNOWN_VENDORS:
        for _ in range(per_vendor):
            gl = _pick_gl(rng, v)
            history.append(_line(
                rng, vendor_name=rng.choice(v.raw_names), gl=gl, amount_range=v.amount_range,
                entity=rng.choice(v.entities), category=InvoiceCategory.ROUTINE,
            ))
    return history


CSV_FIELDS = [
    "invoice_id", "raw_vendor_name", "description", "amount", "currency", "entity_id",
    "cost_centre", "category", "ground_truth_gl", "anomaly",
]


def write_invoices_csv(invoices: list[InvoiceLine], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for inv in invoices:
            writer.writerow({
                "invoice_id": str(inv.invoice_id), "raw_vendor_name": inv.raw_vendor_name,
                "description": inv.description, "amount": inv.amount, "currency": inv.currency,
                "entity_id": inv.entity_id, "cost_centre": inv.cost_centre or "",
                "category": inv.category.value, "ground_truth_gl": inv.ground_truth_gl or "",
                "anomaly": inv.anomaly or "",
            })


def generate_dataset(
    n_routine: int = 700,
    n_near_edge: int = 200,
    n_hard_edge: int = 100,
    output_path: str = "data/invoices.csv",
    vendor_map_path: str = "data/vendor_map.csv",
    seed: int = 42,
) -> list[InvoiceLine]:
    """Generate the synthetic dataset and vendor master, and write both to CSV."""
    invoices, vendor_rows = generate_invoices(n_routine, n_near_edge, n_hard_edge, seed)
    write_invoices_csv(invoices, output_path)
    Path(vendor_map_path).parent.mkdir(parents=True, exist_ok=True)
    with open(vendor_map_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_name", "canonical_id"])
        writer.writeheader()
        writer.writerows(vendor_rows)
    print(f"Generated {len(invoices)} invoices -> {output_path}")
    print(f"Vendor master ({len(vendor_rows)} names) -> {vendor_map_path}")
    return invoices


if __name__ == "__main__":
    generate_dataset()
