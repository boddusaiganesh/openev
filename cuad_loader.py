"""
LexArena — CUAD Loader (Tier 1 Data)
Loads CUAD dataset and converts to LexArena Tier 1 format.
Supports both HuggingFace online loading and local cache.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from lexarena_models import Tier1Sample

# CUAD question categories (all 41 official CUAD types)
CUAD_CATEGORIES = [
    "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period to Terminate Renewal", "Governing Law",
    "Most Favored Nation", "Non-Compete", "Exclusivity", "No-Solicit of Customers",
    "No-Solicit of Employees", "Non-Disparagement", "Termination for Convenience",
    "ROFR/ROFO/ROFN", "Change of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
    "Volume Restriction", "IP Ownership Assignment", "Joint IP Ownership",
    "License Grant", "Non-Transferable License", "Affiliate IP License-Licensor",
    "Affiliate IP License-Licensee", "Unlimited/All-You-Can-Eat License",
    "Irrevocable or Perpetual License", "Source Code Escrow", "Post-Termination Services",
    "Audit Rights", "Uncapped Liability", "Cap on Liability",
    "Liquidated Damages", "Warranty Duration", "Insurance",
    "Covenant not to Sue", "Third Party Beneficiary", "Force Majeure", "Indemnification"
]

# Subset most relevant to LexDomino crisis scenarios
LEXARENA_PRIORITY_CATEGORIES = [
    "Force Majeure", "Indemnification", "Termination for Convenience",
    "Change of Control", "Liquidated Damages", "Insurance",
    "Governing Law", "Cap on Liability", "Uncapped Liability",
    "Audit Rights", "License Grant", "Non-Compete", "Exclusivity",
    "Minimum Commitment", "Warranty Duration"
]


def _extract_question_category(raw_question: str) -> str:
    """Extract the clean category name from a CUAD question string."""
    match = re.search(r'"(.*?)"', raw_question)
    if match:
        return match.group(1)
    return raw_question[:80]


def load_cuad_dataset(
    cache_dir: str = "./cache_dir",
    max_samples: Optional[int] = None,
    categories: Optional[List[str]] = None,
    priority_only: bool = False,
) -> List[Tier1Sample]:
    """
    Load CUAD test set from HuggingFace and convert to Tier1Sample list.
    Falls back to cached JSON if HuggingFace unavailable.
    """
    samples: List[Tier1Sample] = []
    target_categories = categories or (LEXARENA_PRIORITY_CATEGORIES if priority_only else None)

    # Try HuggingFace first
    try:
        import datasets as ds_lib
        cuad = ds_lib.load_dataset(
            "theatticusproject/cuad-qa",
            cache_dir=cache_dir,
        )
        test_set = cuad["test"]
        for item in test_set:
            cat = _extract_question_category(item["question"])
            if target_categories and cat not in target_categories:
                continue
            sample = Tier1Sample(
                sample_id=item["id"],
                contract_name=item["id"].split("__")[0] if "__" in item["id"] else item["id"],
                question_category=cat,
                context=item["context"],
                question=cat,  # Use clean category name as the question
                ground_truth=item["answers"]["text"],
                has_answer=len(item["answers"]["text"]) > 0,
            )
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break
        print(f"[CUAD Loader] Loaded {len(samples)} samples from HuggingFace.")
        return samples

    except Exception as e:
        print(f"[CUAD Loader] HuggingFace unavailable ({e}). Trying local cache...")

    # Fallback: local JSON cache
    local_path = os.path.join(cache_dir, "cuad_tier1_samples.json")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            cat = item.get("question_category", "")
            if target_categories and cat not in target_categories:
                continue
            samples.append(Tier1Sample(**item))
            if max_samples and len(samples) >= max_samples:
                break
        print(f"[CUAD Loader] Loaded {len(samples)} samples from local cache.")
        return samples

    # Final fallback: built-in sample set
    print("[CUAD Loader] No data source found. Using built-in sample set.")
    return _builtin_samples(max_samples, target_categories)


def save_cuad_cache(samples: List[Tier1Sample], cache_dir: str = "./cache_dir") -> str:
    """Save loaded samples to local JSON cache for offline use."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "cuad_tier1_samples.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in samples], f, indent=2)
    print(f"[CUAD Loader] Saved {len(samples)} samples to {path}")
    return path


def get_category_distribution(samples: List[Tier1Sample]) -> Dict[str, int]:
    """Return count of samples per question category."""
    dist: Dict[str, int] = {}
    for s in samples:
        dist[s.question_category] = dist.get(s.question_category, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def _builtin_samples(
    max_samples: Optional[int],
    target_categories: Optional[List[str]],
) -> List[Tier1Sample]:
    """
    Built-in minimal sample set — 15 hand-crafted samples covering
    the LexArena priority categories. Used when CUAD is unavailable.
    """
    raw_samples = [
        {
            "sample_id": "builtin-001",
            "contract_name": "MasterServiceAgreement_TechCorp",
            "question_category": "Force Majeure",
            "context": (
                "Section 14. Force Majeure. Neither party shall be liable for "
                "any failure or delay in performance under this Agreement to the "
                "extent said failures or delays are proximately caused by causes "
                "beyond that party's reasonable control and occurring without its "
                "fault or negligence, including, without limitation, acts of God, "
                "fire, flood, government-mandated trade restrictions, or labor strikes. "
                "The affected party shall provide written notice within five (5) business "
                "days of the occurrence of the force majeure event."
            ),
            "ground_truth": [
                "Neither party shall be liable for any failure or delay in performance "
                "under this Agreement to the extent said failures or delays are proximately "
                "caused by causes beyond that party's reasonable control and occurring "
                "without its fault or negligence, including, without limitation, acts of God, "
                "fire, flood, government-mandated trade restrictions, or labor strikes."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-002",
            "contract_name": "SupplyAgreement_IndustrialCo",
            "question_category": "Liquidated Damages",
            "context": (
                "Section 8. Delivery Obligations. Supplier shall deliver all ordered "
                "goods within fourteen (14) days of the purchase order date. In the event "
                "of late delivery, Supplier shall pay to Buyer liquidated damages in the "
                "amount of $10,000 per week of delay, up to a maximum of $100,000, which "
                "the parties agree represents a reasonable estimate of the harm caused by "
                "late delivery and not a penalty."
            ),
            "ground_truth": [
                "Supplier shall pay to Buyer liquidated damages in the amount of $10,000 "
                "per week of delay, up to a maximum of $100,000."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-003",
            "contract_name": "LoanAgreement_BankCorp",
            "question_category": "Cap on Liability",
            "context": (
                "Section 12. Limitation of Liability. In no event shall either party be "
                "liable to the other for any indirect, incidental, consequential, punitive, "
                "or special damages. The total aggregate liability of either party to the "
                "other under this Agreement shall not exceed the total amounts paid or "
                "payable under this Agreement in the twelve (12) months immediately "
                "preceding the event giving rise to such liability."
            ),
            "ground_truth": [
                "The total aggregate liability of either party to the other under this "
                "Agreement shall not exceed the total amounts paid or payable under this "
                "Agreement in the twelve (12) months immediately preceding the event "
                "giving rise to such liability."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-004",
            "contract_name": "InsurancePolicy_LloydsSyndicate",
            "question_category": "Insurance",
            "context": (
                "Section 5. Coverage. The insurer shall indemnify the insured against "
                "all direct losses arising from trade disruption events up to a maximum "
                "coverage amount of $2,000,000 per occurrence and $5,000,000 in the "
                "aggregate per policy year. Claims must be submitted in writing within "
                "forty-eight (48) hours of the triggering event."
            ),
            "ground_truth": [
                "The insurer shall indemnify the insured against all direct losses arising "
                "from trade disruption events up to a maximum coverage amount of $2,000,000 "
                "per occurrence and $5,000,000 in the aggregate per policy year."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-005",
            "contract_name": "EmploymentAgreement_SeniorExec",
            "question_category": "Change of Control",
            "context": (
                "Section 9. Change of Control. In the event of a Change of Control of "
                "the Company, all unvested equity awards held by Employee shall immediately "
                "vest and become exercisable. For purposes of this Agreement, 'Change of "
                "Control' means the acquisition by any person or group of more than fifty "
                "percent (50%) of the outstanding voting securities of the Company."
            ),
            "ground_truth": [
                "In the event of a Change of Control of the Company, all unvested equity "
                "awards held by Employee shall immediately vest and become exercisable."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-006",
            "contract_name": "SoftwareLicenseAgreement_SaaSCo",
            "question_category": "License Grant",
            "context": (
                "Section 2. License Grant. Subject to the terms and conditions of this "
                "Agreement and payment of all applicable fees, Licensor hereby grants to "
                "Licensee a non-exclusive, non-transferable, non-sublicensable, "
                "worldwide license to use the Software solely for Licensee's internal "
                "business purposes during the Term."
            ),
            "ground_truth": [
                "Licensor hereby grants to Licensee a non-exclusive, non-transferable, "
                "non-sublicensable, worldwide license to use the Software solely for "
                "Licensee's internal business purposes during the Term."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-007",
            "contract_name": "DistributionAgreement_RetailCo",
            "question_category": "Exclusivity",
            "context": (
                "Section 4. Distribution Rights. During the Term of this Agreement, "
                "Supplier grants to Distributor the exclusive right to distribute, market, "
                "and sell the Products within the Territory. Supplier shall not appoint "
                "any other distributor for the Products in the Territory and shall refer "
                "all customer inquiries from the Territory exclusively to Distributor."
            ),
            "ground_truth": [
                "Supplier grants to Distributor the exclusive right to distribute, market, "
                "and sell the Products within the Territory."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-008",
            "contract_name": "ConstructionContract_BuilderCorp",
            "question_category": "Indemnification",
            "context": (
                "Section 11. Indemnification. Contractor shall defend, indemnify, and hold "
                "harmless Owner, its officers, directors, employees, and agents from and "
                "against any and all claims, damages, losses, costs, and expenses, including "
                "reasonable attorneys' fees, arising out of or related to Contractor's "
                "performance of the Work, to the extent caused by the negligent acts or "
                "omissions of Contractor, its subcontractors, or their respective employees."
            ),
            "ground_truth": [
                "Contractor shall defend, indemnify, and hold harmless Owner, its officers, "
                "directors, employees, and agents from and against any and all claims, "
                "damages, losses, costs, and expenses, including reasonable attorneys' fees, "
                "arising out of or related to Contractor's performance of the Work."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-009",
            "contract_name": "NDAAgreement_Startup",
            "question_category": "Non-Compete",
            "context": (
                "Section 3. General Provisions. This Agreement does not restrict either "
                "party from engaging in any business activities, entering into any "
                "commercial relationships, or competing with the other party in any "
                "market or industry. The parties acknowledge that this Agreement is "
                "limited solely to the protection of Confidential Information."
            ),
            "ground_truth": [],
            "has_answer": False,
        },
        {
            "sample_id": "builtin-010",
            "contract_name": "LeaseAgreement_PropertyCo",
            "question_category": "Termination for Convenience",
            "context": (
                "Section 18. Termination. Either party may terminate this Agreement "
                "without cause upon ninety (90) days' prior written notice to the other "
                "party. In the event of termination without cause by Landlord, Landlord "
                "shall pay Tenant a termination fee equal to six (6) months of the then-"
                "current monthly base rent."
            ),
            "ground_truth": [
                "Either party may terminate this Agreement without cause upon ninety (90) "
                "days' prior written notice to the other party."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-011",
            "contract_name": "AerospaceSupply_VanguardCo",
            "question_category": "Force Majeure",
            "context": (
                "Section 14.2. Force Majeure Events. For the purposes of this clause, "
                "Force Majeure Events include government-mandated export restrictions, "
                "sanctions, embargoes, acts of war, terrorism, pandemic, or natural "
                "disaster. Notice must be given within five (5) business days. Failure "
                "to provide timely notice shall be deemed a waiver of Force Majeure rights."
            ),
            "ground_truth": [
                "Notice must be given within five (5) business days.",
                "Failure to provide timely notice shall be deemed a waiver of Force Majeure rights."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-012",
            "contract_name": "PharmaceuticalLicense_HelixBio",
            "question_category": "Audit Rights",
            "context": (
                "Section 7. Records and Audit. Licensee shall maintain complete and "
                "accurate books of account relating to the royalties payable under this "
                "Agreement. Licensor shall have the right, upon thirty (30) days' written "
                "notice, to audit such books and records no more than once per calendar "
                "year at Licensor's expense, unless the audit reveals an underpayment "
                "of more than five percent (5%), in which case the audit costs shall be "
                "borne by Licensee."
            ),
            "ground_truth": [
                "Licensor shall have the right, upon thirty (30) days' written notice, "
                "to audit such books and records no more than once per calendar year at "
                "Licensor's expense."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-013",
            "contract_name": "CreditFacility_PacificCapital",
            "question_category": "Minimum Commitment",
            "context": (
                "Section 4. Financial Covenants. Borrower shall maintain at all times "
                "a minimum cash balance of not less than One Million Five Hundred Thousand "
                "Dollars ($1,500,000). Borrower's failure to maintain the minimum cash "
                "balance at any time during the term of this facility shall constitute "
                "an Event of Default."
            ),
            "ground_truth": [
                "Borrower shall maintain at all times a minimum cash balance of not less "
                "than One Million Five Hundred Thousand Dollars ($1,500,000)."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-014",
            "contract_name": "DataProcessingAgreement_CloudCo",
            "question_category": "Warranty Duration",
            "context": (
                "Section 6. Warranties. Each party represents and warrants that it has "
                "the full power and authority to enter into this Agreement. The Service "
                "Provider additionally warrants that the Services will conform to the "
                "specifications set forth in the applicable Statement of Work for a period "
                "of ninety (90) days from the date of delivery."
            ),
            "ground_truth": [
                "The Service Provider additionally warrants that the Services will conform "
                "to the specifications set forth in the applicable Statement of Work for a "
                "period of ninety (90) days from the date of delivery."
            ],
            "has_answer": True,
        },
        {
            "sample_id": "builtin-015",
            "contract_name": "MergerAgreement_AcquiCo",
            "question_category": "Governing Law",
            "context": (
                "Section 22. Miscellaneous. This Agreement shall be governed by and "
                "construed in accordance with the laws of the State of Delaware, without "
                "giving effect to any choice or conflict of law provision or rule. Any "
                "legal suit, action, or proceeding arising out of or related to this "
                "Agreement shall be instituted exclusively in the federal courts of "
                "the United States or the courts of the State of Delaware."
            ),
            "ground_truth": [
                "This Agreement shall be governed by and construed in accordance with the "
                "laws of the State of Delaware."
            ],
            "has_answer": True,
        },
    ]

    results = []
    for item in raw_samples:
        cat = item["question_category"]
        if target_categories and cat not in target_categories:
            continue
        # Ensure question field is set (derived from question_category if absent)
        if "question" not in item:
            item = {**item, "question": cat}
        results.append(Tier1Sample(**item))
        if max_samples and len(results) >= max_samples:
            break

    print(f"[CUAD Loader] Using built-in sample set: {len(results)} samples.")
    return results
