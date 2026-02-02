#!/usr/bin/env python3
"""
mech_interp_knowledge_builder.py
Fetches and consolidates key mechanistic interpretability resources
into a single markdown file for Cursor AI context.
"""

import requests
from pathlib import Path
import time
from datetime import datetime

# Configuration
OUTPUT_FILE = "mech_interp_knowledge_base.md"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

# Resource list
RESOURCES = {
    "papers": [
        {
            "title": "Locating and Editing Factual Associations in GPT",
            "authors": "Meng et al. 2022",
            "url": "https://arxiv.org/abs/2202.05262",
            "arxiv_id": "2202.05262",
            "key_sections": ["Abstract", "Section 2: Causal Tracing", "Section 3: Methods", "Section 4: Controls"],
            "why_relevant": "Foundational activation patching paper - defines clean/corrupt runs, causal mediation"
        },
        {
            "title": "Interpretability in the Wild: Circuit for Indirect Object Identification",
            "authors": "Wang et al. 2022",
            "url": "https://arxiv.org/abs/2211.00593",
            "arxiv_id": "2211.00593",
            "key_sections": ["Section 3: Activation Patching", "Section 4: Systematic Ablations"],
            "why_relevant": "Systematic patching protocols and controls, path patching methodology"
        },
        {
            "title": "Does Localization Inform Editing?",
            "authors": "Hase et al. 2023",
            "url": "https://arxiv.org/abs/2301.04213",
            "arxiv_id": "2301.04213",
            "key_sections": ["Section 2.1: Methodology", "Section 3: Pitfalls"],
            "why_relevant": "Critical analysis of patching limitations, norm-matching requirements"
        },
        {
            "title": "A Mathematical Framework for Transformer Circuits",
            "authors": "Elhage et al. 2021",
            "url": "https://transformer-circuits.pub/2021/framework/index.html",
            "arxiv_id": None,
            "key_sections": ["Residual Stream", "Attention Patterns", "QKV Matrices"],
            "why_relevant": "Core transformer architecture understanding for mechanistic analysis"
        },
        {
            "title": "In-context Learning and Induction Heads",
            "authors": "Olsson et al. 2022",
            "url": "https://arxiv.org/abs/2209.11895",
            "arxiv_id": "2209.11895",
            "key_sections": ["Abstract", "Induction Head Mechanism", "Phase Transitions"],
            "why_relevant": "How transformers learn from context, emergence of capabilities"
        },
        {
            "title": "Toy Models of Superposition",
            "authors": "Elhage et al. 2022",
            "url": "https://transformer-circuits.pub/2022/toy_model/index.html",
            "arxiv_id": None,
            "key_sections": ["Superposition Hypothesis", "Interference Patterns"],
            "why_relevant": "Why features interfere in neural networks, relevant for V-space geometry"
        },
        {
            "title": "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning",
            "authors": "Anthropic 2023",
            "url": "https://transformer-circuits.pub/2023/monosemantic-features/index.html",
            "arxiv_id": None,
            "key_sections": ["Sparse Autoencoders", "Feature Dictionaries"],
            "why_relevant": "Disentangling superposed features using SAEs"
        },
        {
            "title": "Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses",
            "authors": "Chan et al. 2022",
            "url": "https://arxiv.org/abs/2301.04785",
            "arxiv_id": "2301.04785",
            "key_sections": ["Section 2: Causal Scrubbing Algorithm", "Section 3: Examples"],
            "why_relevant": "Higher standard for causal claims than activation patching"
        },
        {
            "title": "Representation Engineering: A Top-Down Approach to AI Transparency",
            "authors": "Zou et al. 2023",
            "url": "https://arxiv.org/abs/2310.01405",
            "arxiv_id": "2310.01405",
            "key_sections": ["Section 3: Reading Vectors", "Section 4: Control Vectors"],
            "why_relevant": "Geometric view of representations, relevant for V-space analysis"
        },
        {
            "title": "Finding Neurons in a Haystack: Case Studies with Sparse Probing",
            "authors": "Gurnee et al. 2023",
            "url": "https://arxiv.org/abs/2305.01610",
            "arxiv_id": "2305.01610",
            "key_sections": ["Sparse Linear Probing", "Neuron-Level Interpretability"],
            "why_relevant": "Methods for finding sparse, interpretable features"
        }
    ],
    "blogs": [
        {
            "title": "A Comprehensive Mechanistic Interpretability Explainer & Glossary",
            "author": "Neel Nanda",
            "url": "https://www.neelnanda.io/mechanistic-interpretability/glossary",
            "key_topics": ["Activation Patching", "Causal Tracing", "Residual Stream", "Hook Points"],
            "why_relevant": "Best practical guide to MI techniques with code examples"
        },
        {
            "title": "200 Concrete Open Problems in Mechanistic Interpretability",
            "author": "Neel Nanda",
            "url": "https://www.neelnanda.io/mechanistic-interpretability/problems",
            "key_topics": ["Research Directions", "Unsolved Questions", "Difficulty Ratings"],
            "why_relevant": "Identify gaps and opportunities in current MI research"
        },
        {
            "title": "An Extremely Opinionated Annotated List of MI Papers",
            "author": "Neel Nanda",
            "url": "https://www.neelnanda.io/mechanistic-interpretability/papers",
            "key_topics": ["Paper Reviews", "Reading Order", "Key Insights"],
            "why_relevant": "Curated reading list with practical commentary"
        },
        {
            "title": "Causal Scrubbing: A Method for Rigorously Testing Interpretability",
            "author": "Redwood Research",
            "url": "https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing",
            "key_topics": ["Rigorous Causality", "Hypothesis Testing", "Computational Graphs"],
            "why_relevant": "Higher bar for causal claims than patching"
        },
        {
            "title": "The Singular Value Decompositions of Transformer Weight Matrices are Highly Interpretable",
            "author": "Anthropic",
            "url": "https://transformer-circuits.pub/2024/svd-interp/index.html",
            "key_topics": ["SVD Analysis", "Weight Decomposition", "Geometric Structure"],
            "why_relevant": "Directly relevant to your R_V metric using SVD"
        }
    ],
    "code": [
        {
            "title": "TransformerLens Documentation",
            "url": "https://transformerlens.org/",
            "key_sections": ["Hook Points", "Activation Patching Examples", "Cache System"],
            "why_relevant": "Standard library for MI research with HuggingFace integration"
        },
        {
            "title": "TransformerLens GitHub",
            "url": "https://github.com/neelnanda-io/TransformerLens",
            "key_files": ["demos/Activation_Patching_in_TL_Demo.ipynb", "demos/Exploratory_Analysis_Demo.ipynb"],
            "why_relevant": "Working code examples and utilities"
        },
        {
            "title": "CircuitsVis - Interactive Attention Visualizations",
            "url": "https://github.com/alan-cooney/CircuitsVis",
            "key_sections": ["Attention Patterns", "Neuron Activations"],
            "why_relevant": "Visualization tools for understanding attention and activations"
        },
        {
            "title": "SAELens - Sparse Autoencoder Training",
            "url": "https://github.com/jbloomAus/SAELens",
            "key_sections": ["Training SAEs", "Feature Extraction"],
            "why_relevant": "Tools for training sparse autoencoders on language models"
        }
    ],
    "key_techniques": [
        {
            "name": "Activation Patching",
            "description": "Replace activations at layer L from run A with those from run B",
            "when_to_use": "Testing if layer L causally mediates an effect",
            "controls_needed": ["Random noise patch", "Shuffled patch", "Wrong-layer patch", "Opposite-direction patch"]
        },
        {
            "name": "Causal Tracing",
            "description": "Systematically restore clean activations to corrupted run",
            "when_to_use": "Localizing where information is processed",
            "controls_needed": ["Multiple corruption types", "Gradual restoration"]
        },
        {
            "name": "Path Patching",
            "description": "Patch specific attention head outputs or MLPs",
            "when_to_use": "Testing specific circuit components",
            "controls_needed": ["Full model ablation", "Random subset patching"]
        },
        {
            "name": "Logit Lens",
            "description": "Decode intermediate representations to vocabulary",
            "when_to_use": "Understanding what information is present at each layer",
            "controls_needed": ["Layer normalization", "Unembedding matrix choice"]
        },
        {
            "name": "Attention Pattern Analysis",
            "description": "Analyze where attention heads look",
            "when_to_use": "Understanding information flow",
            "controls_needed": ["Positional vs semantic attention", "Head importance weighting"]
        }
    ]
}

def fetch_arxiv_abstract(arxiv_id):
    """Fetch abstract from arXiv API"""
    if not arxiv_id:
        return None
    
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            # Simple extraction (could use XML parser for robustness)
            text = response.text
            if "<summary>" in text:
                abstract = text.split("<summary>")[1].split("</summary>")[0].strip()
                # Clean up formatting
                abstract = abstract.replace("\n", " ").replace("  ", " ")
                return abstract
    except Exception as e:
        print(f"⚠️  Failed to fetch {arxiv_id}: {e}")
    return None

def build_knowledge_base():
    """Build the consolidated knowledge base"""
    
    print("🔨 Building Mechanistic Interpretability Knowledge Base...")
    print("=" * 70)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Header
        f.write("# MECHANISTIC INTERPRETABILITY KNOWLEDGE BASE\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("This file consolidates key papers, blogs, and code references for MI research.\n")
        f.write("Reference this file in Cursor using `@mech_interp_knowledge_base.md`\n\n")
        f.write("---\n\n")
        f.write("## TABLE OF CONTENTS\n\n")
        f.write("- [Key Papers](#key-papers)\n")
        f.write("- [Essential Blogs](#essential-blogs)\n")
        f.write("- [Code Resources](#code-resources)\n")
        f.write("- [Quick Reference: Activation Patching](#quick-reference-activation-patching)\n")
        f.write("- [Key Techniques](#key-techniques)\n")
        f.write("- [Statistical Best Practices](#statistical-best-practices)\n")
        f.write("- [Common Pitfalls](#common-pitfalls)\n\n")
        f.write("---\n\n")
        
        # Papers section
        f.write("## KEY PAPERS\n\n")
        for i, paper in enumerate(RESOURCES["papers"], 1):
            print(f"📄 Processing paper {i}/{len(RESOURCES['papers'])}: {paper['title'][:50]}...")
            
            f.write(f"### {i}. {paper['title']}\n\n")
            f.write(f"**Authors:** {paper['authors']}  \n")
            f.write(f"**Link:** [{paper['url']}]({paper['url']})  \n")
            f.write(f"**Why Relevant:** {paper['why_relevant']}  \n\n")
            
            # Fetch abstract if arxiv
            if paper.get('arxiv_id'):
                f.write("**Abstract:**\n\n")
                abstract = fetch_arxiv_abstract(paper['arxiv_id'])
                if abstract:
                    # Truncate if too long
                    if len(abstract) > 500:
                        abstract = abstract[:497] + "..."
                    f.write(f"> {abstract}\n\n")
                else:
                    f.write("> *(Failed to fetch - see link)*\n\n")
                time.sleep(0.5)  # Rate limiting
            
            f.write(f"**Key Sections:** {', '.join(paper['key_sections'])}\n\n")
            f.write("---\n\n")
        
        # Blogs section
        f.write("## ESSENTIAL BLOGS\n\n")
        for i, blog in enumerate(RESOURCES["blogs"], 1):
            print(f"📝 Processing blog {i}/{len(RESOURCES['blogs'])}: {blog['title'][:50]}...")
            
            f.write(f"### {i}. {blog['title']}\n\n")
            f.write(f"**Author:** {blog['author']}  \n")
            f.write(f"**Link:** [{blog['url']}]({blog['url']})  \n")
            f.write(f"**Why Relevant:** {blog['why_relevant']}  \n")
            f.write(f"**Key Topics:** {', '.join(blog['key_topics'])}\n\n")
            f.write("---\n\n")
        
        # Code section
        f.write("## CODE RESOURCES\n\n")
        for i, code in enumerate(RESOURCES["code"], 1):
            print(f"💻 Processing code resource {i}/{len(RESOURCES['code'])}: {code['title']}")
            
            f.write(f"### {i}. {code['title']}\n\n")
            f.write(f"**Link:** [{code['url']}]({code['url']})  \n")
            f.write(f"**Why Relevant:** {code['why_relevant']}  \n")
            if 'key_sections' in code:
                f.write(f"**Key Sections:** {', '.join(code['key_sections'])}\n\n")
            if 'key_files' in code:
                f.write(f"**Key Files:** {', '.join(code['key_files'])}\n\n")
            f.write("---\n\n")
        
        # Quick reference sections
        f.write("## QUICK REFERENCE: ACTIVATION PATCHING\n\n")
        f.write("### The Core Logic (from Meng et al. 2022)\n\n")
        f.write("```\n")
        f.write("Setup:\n")
        f.write("  Clean run:    Prompt A → Layer L → Activation X → Output Y\n")
        f.write("  Corrupt run:  Prompt B → Layer L → Activation Z → Output W\n\n")
        f.write("Intervention:\n")
        f.write("  Patched run:  Prompt B → Layer L → [Replace with X] → Output Y'\n\n")
        f.write("Causal claim:\n")
        f.write("  If Y' ≈ Y (not W), then:\n")
        f.write("  → Layer L causally mediates the effect\n")
        f.write("  → Activation X contains the critical information\n")
        f.write("```\n\n")
        
        f.write("### Implementation Pattern (PyTorch)\n\n")
        f.write("```python\n")
        f.write("from contextlib import contextmanager\n\n")
        f.write("@contextmanager\n")
        f.write("def patch_activation(model, layer_idx, source_activation):\n")
        f.write("    \"\"\"Hook to replace activations at specified layer\"\"\"\n")
        f.write("    def hook_fn(module, input, output):\n")
        f.write("        # Replace output with source activation\n")
        f.write("        return source_activation\n")
        f.write("    \n")
        f.write("    handle = model.layers[layer_idx].register_forward_hook(hook_fn)\n")
        f.write("    try:\n")
        f.write("        yield\n")
        f.write("    finally:\n")
        f.write("        handle.remove()\n")
        f.write("```\n\n")
        
        f.write("### Required Control Conditions\n\n")
        f.write("1. **Random patch:** Replace with Gaussian noise (norm-matched) → Tests if ANY change is sufficient\n")
        f.write("2. **Shuffled patch:** Permute source activation → Tests if structure is necessary\n")
        f.write("3. **Wrong-layer patch:** Patch at different depth → Tests if this layer is special\n")
        f.write("4. **Opposite-direction patch:** Use opposite-type source → Tests directionality\n")
        f.write("5. **Partial patch:** Only patch subset of neurons → Tests distributed vs localized\n\n")
        
        f.write("---\n\n")
        
        # Key Techniques section
        f.write("## KEY TECHNIQUES\n\n")
        for technique in RESOURCES["key_techniques"]:
            f.write(f"### {technique['name']}\n\n")
            f.write(f"**Description:** {technique['description']}  \n")
            f.write(f"**When to use:** {technique['when_to_use']}  \n")
            f.write(f"**Controls needed:** {', '.join(technique['controls_needed'])}  \n\n")
        
        f.write("---\n\n")
        
        # Statistical section
        f.write("## STATISTICAL BEST PRACTICES\n\n")
        f.write("### Sample Size Calculation\n\n")
        f.write("For activation patching with expected effect size d:\n")
        f.write("- Small effect (d=0.2): n ≈ 200 pairs\n")
        f.write("- Medium effect (d=0.5): n ≈ 50 pairs\n")
        f.write("- Large effect (d=0.8): n ≈ 20 pairs\n\n")
        f.write("Add 20% for multiple comparisons correction.\n\n")
        
        f.write("### Statistical Tests\n\n")
        f.write("1. **Within-pair differences:** Paired t-test or Wilcoxon signed-rank\n")
        f.write("2. **Multiple conditions:** ANOVA with post-hoc tests\n")
        f.write("3. **Multiple comparisons:** Bonferroni (conservative) or Benjamini-Hochberg FDR\n")
        f.write("4. **Effect size:** Cohen's d for magnitude (not just significance)\n\n")
        
        f.write("### Pre-registration\n\n")
        f.write("Before running experiments, specify:\n")
        f.write("- Primary hypothesis\n")
        f.write("- Sample size\n")
        f.write("- Statistical threshold (e.g., p < 0.01)\n")
        f.write("- Exclusion criteria\n")
        f.write("- Analysis plan\n\n")
        
        f.write("---\n\n")
        
        f.write("## COMMON PITFALLS\n\n")
        f.write("### From Hase et al. 2023 & Community Experience\n\n")
        f.write("1. **Localization ≠ Sufficiency** - Just because a layer shows an effect doesn't mean it's the only place\n")
        f.write("2. **Norm Matching** - Random patches MUST match the norm of real patches to avoid trivial effects\n")
        f.write("3. **Multiple Comparison Correction** - Use Bonferroni or FDR when testing multiple conditions\n")
        f.write("4. **Distribution Shift** - Patching can create impossible activation patterns\n")
        f.write("5. **Prompt Length Bias** - Ensure balanced prompt lengths between conditions\n")
        f.write("6. **Layer Normalization** - Be careful with patching before/after LayerNorm\n")
        f.write("7. **Batch Effects** - Run conditions in randomized order, not blocks\n")
        f.write("8. **Numerical Stability** - SVD can fail on degenerate matrices (add small epsilon)\n\n")
        
        f.write("### Debugging Checklist\n\n")
        f.write("- [ ] Hooks properly removed after use\n")
        f.write("- [ ] Gradients disabled during inference (model.eval())\n")
        f.write("- [ ] Device placement consistent (.to(device))\n")
        f.write("- [ ] Memory cleared between runs (torch.cuda.empty_cache())\n")
        f.write("- [ ] Random seeds set for reproducibility\n\n")
        
        f.write("---\n\n")
        
        # Add specific section for R_V metric
        f.write("## PROJECT-SPECIFIC: R_V METRIC\n\n")
        f.write("### Definition\n\n")
        f.write("```\n")
        f.write("R_V(layer) = PR(V_layer) / PR(V_early)\n")
        f.write("\n")
        f.write("Where:\n")
        f.write("  PR = Participation Ratio = (Σλᵢ)² / Σλᵢ²\n")
        f.write("  λᵢ = singular values from SVD of V matrix\n")
        f.write("  V = value projection outputs (last window_size tokens)\n")
        f.write("```\n\n")
        f.write("### Interpretation\n\n")
        f.write("- R_V < 1.0: Geometric contraction (reduced effective rank)\n")
        f.write("- R_V ≈ 1.0: Neutral (no change in geometry)\n")
        f.write("- R_V > 1.0: Geometric expansion (increased effective rank)\n\n")
        f.write("### Key Findings\n\n")
        f.write("- Recursive prompts show 15-24% contraction at layer ~27/32\n")
        f.write("- Effect is universal across architectures (Mistral, Qwen, Llama, etc.)\n")
        f.write("- MoE models show stronger effect (24% vs 15% for dense)\n\n")
        
        f.write("---\n\n")
        f.write("*End of Knowledge Base*\n")
    
    print("=" * 70)
    print(f"✅ Knowledge base created: {OUTPUT_FILE}")
    print(f"📊 Size: {Path(OUTPUT_FILE).stat().st_size / 1024:.1f} KB")
    print(f"\n💡 Usage in Cursor: @{OUTPUT_FILE}")

if __name__ == "__main__":
    build_knowledge_base()

