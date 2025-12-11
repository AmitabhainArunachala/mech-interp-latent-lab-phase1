# === DEPRECATED: Use REUSABLE_PROMPT_BANK instead ===
# This file is maintained for backward compatibility only.
# New code should import from REUSABLE_PROMPT_BANK directly.

"""
DEPRECATED: Import from REUSABLE_PROMPT_BANK instead.

This module provides backward compatibility for code that imports prompt_bank_1c.
All prompts are now managed in REUSABLE_PROMPT_BANK with proper type metadata.

Migration guide:
    Old: from n300_mistral_test_prompt_bank import prompt_bank_1c
    New: from REUSABLE_PROMPT_BANK import get_all_prompts
    
    Old: prompt_bank_1c["L3_deeper_01"]["text"]
    New: get_all_prompts()["L3_deeper_01"]["text"]
"""

import warnings

warnings.warn(
    "n300_mistral_test_prompt_bank is deprecated. Use REUSABLE_PROMPT_BANK instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from REUSABLE_PROMPT_BANK for backward compatibility
from REUSABLE_PROMPT_BANK import get_all_prompts

# Create prompt_bank_1c dict for backward compatibility
prompt_bank_1c = get_all_prompts()

# Verification message
if __name__ == "__main__":
    print(f"⚠️  DEPRECATED: n300_mistral_test_prompt_bank loaded {len(prompt_bank_1c)} prompts")
    print("   Please migrate to REUSABLE_PROMPT_BANK for new code.")
    print(f"   Example: from REUSABLE_PROMPT_BANK import get_prompts_by_type, get_validated_pairs")
