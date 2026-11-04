# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""
Shared taxonomy definitions for GuardrailTrinity data pipelines.

Single source of truth for all category mappings used by both
generate_train_datasets.py and generate_test_datasets.py.

Symbols exported:
  AEGIS_MAP          — raw name→code mapping (23 categories, S1–S23)
  AEGIS_TAXONOMY     — code→description dict derived from AEGIS_MAP
  WILD_MAP           — raw key→code mapping (14 categories, W1–W14)
  WILD_TAXONOMY      — code→description dict derived from WILD_MAP
  NOVEL_TAXONOMY     — held-out 6-category schema used by BeaverTails-PolicyShift (N1–N6)
  BEAVER_TO_NOVEL    — mapping from BeaverTails fine-grained keys to NOVEL codes
  SEMANTIC_CLUSTERS  — sets of semantically overlapping descriptions across both
                       taxonomies; used by the procedural taxonomy generator to
                       prevent near-duplicate distractors
  MASTER_DESC        — merged AEGIS ∪ WildGuard description pool (prefixed A_/W_)
                       used for distractor sampling during procedural taxonomy generation
"""

# ==============================================================================
# AEGIS (NVIDIA Aegis-AI-Content-Safety-Dataset-2.0) — 23 categories
# ==============================================================================

AEGIS_MAP = {
    'Violence': 'S1', 'Sexual': 'S2', 'Criminal Planning/Confessions': 'S3',
    'Guns and Illegal Weapons': 'S4', 'Controlled/Regulated Substances': 'S5',
    'Suicide and Self Harm': 'S6', 'Sexual (minor)': 'S7', 'Hate/Identity Hate': 'S8',
    'PII/Privacy': 'S9', 'Harassment': 'S10', 'Threat': 'S11', 'Profanity': 'S12',
    'Needs Caution': 'S13', 'Other': 'S14', 'Manipulation': 'S15',
    'Fraud/Deception': 'S16', 'Malware': 'S17', 'High Risk Gov Decision Making': 'S18',
    'Political/Misinformation/Conspiracy': 'S19', 'Copyright/Trademark/Plagiarism': 'S20',
    'Unauthorized Advice': 'S21', 'Illegal Activity': 'S22', 'Immoral/Unethical': 'S23'
}

AEGIS_TAXONOMY = {v: k for k, v in AEGIS_MAP.items()}

# ==============================================================================
# WildGuard (AllenAI WildGuardMix) — 14 categories
# ==============================================================================

WILD_MAP = {
    'sexual_content': 'W1', 'violence_and_physical_harm': 'W2',
    'toxic_language_hate_speech': 'W3', 'social_stereotypes_and_unfair_discrimination': 'W4',
    'cyberattack': 'W5', 'fraud_assisting_illegal_activities': 'W6',
    'copyright_violations': 'W7', 'private_information_individual': 'W8',
    'causing_material_harm_by_disseminating_misinformation': 'W9',
    'disseminating_false_or_misleading_information_encouraging_disinformation_campaigns': 'W10',
    'sensitive_information_organization_government': 'W11',
    'defamation_encouraging_unethical_or_unsafe_actions': 'W12',
    'mental_health_over-reliance_crisis': 'W13', 'others': 'W14'
}

WILD_TAXONOMY = {v: k.replace('_', ' ').title() for k, v in WILD_MAP.items()}

# ==============================================================================
# Held-out taxonomy — used by BeaverTails-PolicyShift evaluation benchmark
# ==============================================================================

NOVEL_TAXONOMY = {
    'N1': 'Physical Aggression and Injury', 'N2': 'Identity-Based Hostility',
    'N3': 'Adult and Obscene Content', 'N4': 'Deception and Financial Malpractice',
    'N5': 'Restricted Goods and Criminal Assistance', 'N6': 'Data Privacy and Network Security'
}

BEAVER_TO_NOVEL = {
    "violence,aiding_and_abetting,incitement": "N1", "self_harm": "N1",
    "animal_abuse": "N1", "child_abuse": "N1", "terrorism,organized_crime": "N1",
    "hate_speech,offensive_language": "N2", "discrimination,stereotype,injustice": "N2",
    "controversial_topics,politics": "N2", "sexually_explicit,adult_content": "N3",
    "financial_crime,property_crime,theft": "N4", "misinformation_regarding_ethics,laws_and_safety": "N4",
    "non_violent_unethical_behavior": "N4", "drug_abuse,weapons,banned_substance": "N5",
    "privacy_violation": "N6"
}

# ==============================================================================
# Training pipeline — procedural taxonomy generation helpers
# ==============================================================================

# Groups of descriptions that are semantically equivalent across AEGIS and
# WildGuard. Distractors drawn from the same cluster as a true violation are
# excluded, so the model cannot rely on near-duplicate category names.
SEMANTIC_CLUSTERS = [
    {"Violence", "Violence And Physical Harm"},
    {"Sexual", "Sexual (minor)", "Sexual Content"},
    {"Hate/Identity Hate", "Toxic Language Hate Speech", "Social Stereotypes And Unfair Discrimination", "Harassment"},
    {"Fraud/Deception", "Fraud Assisting Illegal Activities", "Illegal Activity"},
    {"Copyright/Trademark/Plagiarism", "Copyright Violations"},
    {"PII/Privacy", "Private Information Individual", "Sensitive Information Organization Government"},
    {"Political/Misinformation/Conspiracy", "Causing Material Harm By Disseminating Misinformation", "Disseminating False Or Misleading Information Encouraging Disinformation Campaigns"},
    {"Suicide and Self Harm", "Mental Health Over-Reliance Crisis"},
    {"Malware", "Cyberattack"},
    {"Other", "Others"}
]

# Merged description pool (AEGIS ∪ WildGuard) with A_/W_ prefixes to keep
# entries unique even if two descriptions happen to share the same text.
MASTER_DESC = {
    **{f"A_{k}": v for k, v in AEGIS_TAXONOMY.items()},
    **{f"W_{k}": v for k, v in WILD_TAXONOMY.items()},
}
