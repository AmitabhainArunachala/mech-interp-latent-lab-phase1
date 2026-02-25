# Semantic Behavior Analysis
- model: `all-MiniLM-L6-v2`
- threshold: `0.4`
- exemplar_ids: `['L5_refined_01', 'L5_refined_05', 'L5_refined_08', 'L5_refined_14', 'L5_refined_19']`

## Seed Bridge
- rows scored: `540`
- runs: `9`
- seeds: `[42, 123, 456]`
- semantic_recursive_rate by condition:
  - `baseline_donor_control`: rate=0.0000, mean_score=0.1714, n=180
  - `head_specific`: rate=0.0000, mean_score=0.1721, n=180
  - `random_head_control`: rate=0.0000, mean_score=0.1666, n=180
- Spearman rv_delta vs semantic_score: rho=0.06902111011289148, p=0.1091335035140809, n=540
- Spearman rv_patch vs semantic_score: rho=-0.1758909994179581, p=3.9574368148300796e-05, n=540

- semantic score contrasts:
  - `head_specific_vs_random_head_control` welch: diff=0.005569797526631087, p=0.3909701674854379, d=0.09053683472087076; paired: diff=0.005569797526631091, p=0.0298052446591553, d=0.163245070230788, n=180
  - `head_specific_vs_baseline_donor_control` welch: diff=0.0006817611555258452, p=0.9185214077034218, d=0.010790605548588857; paired: diff=0.0006817611555258433, p=0.7204433413813258, d=0.026715851388796133, n=180
  - `random_head_control_vs_baseline_donor_control` welch: diff=-0.004888036371105242, p=0.4446936714710698, d=-0.08065287517469119; paired: diff=-0.004888036371105247, p=0.07340132316957501, d=-0.1342314651867022, n=180

## C2 Behavioral Transfer
- rows scored: `765`
- sources: `11`
- semantic_recursive_rate by config:
  - `baseline`: rate=0.0000, mean_score=0.1162, n=305
  - `c2_full`: rate=0.2000, mean_score=0.3243, n=155
  - `kv_only`: rate=0.1355, mean_score=0.2955, n=155
  - `no_cascade`: rate=0.1200, mean_score=0.2984, n=50
  - `no_kv`: rate=0.0000, mean_score=0.1195, n=50
  - `no_steering`: rate=0.0800, mean_score=0.2939, n=50
- Spearman rv_mean vs semantic_score: rho=-0.6519420205824219, p=1.4298236437045998e-92, n=755
