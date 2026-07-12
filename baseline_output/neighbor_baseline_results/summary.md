# Neighbor Baseline Results Summary

## Methods

| Method | Description |
|--------|-------------|
| majority_gene | Per-gene majority vote (ignores perturbation) |
| majority_pert | Per-perturbation majority vote (ignores target gene) |
| pert_neighbor_* | Find similar perturbations, vote on target gene's response |
| gene_neighbor_* | Find similar target genes, vote on their response |
| both_neighbor_* | Combine perturbation and gene neighbors |
| ensemble | Combine all neighbor sources (ESM2, GenePT, co-pathway) |
| two_stage_* | First predict DE/not_de, then up/down |

## Results by Dataset

### ADAMSON

| Method | Accuracy | Macro F1 | up F1 | down F1 | not_de F1 |
|--------|----------|----------|-------|---------|----------|
| majority_gene | 0.8842 | 0.7734 | 0.6386 | 0.7487 | 0.9330 |
| majority_pert | 0.7851 | 0.2932 | 0.0000 | 0.0000 | 0.8796 |
| pert_neighbor_esm2 | 0.8831 | 0.7729 | 0.6391 | 0.7475 | 0.9323 |
| pert_neighbor_genept | 0.8814 | 0.7699 | 0.6324 | 0.7461 | 0.9311 |
| pert_neighbor_copath | 0.8804 | 0.7671 | 0.6284 | 0.7423 | 0.9306 |
| gene_neighbor_esm2 | 0.8842 | 0.7734 | 0.6386 | 0.7487 | 0.9330 |
| gene_neighbor_genept | 0.8842 | 0.7734 | 0.6386 | 0.7487 | 0.9330 |
| gene_neighbor_copath | 0.8842 | 0.7734 | 0.6386 | 0.7487 | 0.9330 |
| both_neighbor_esm2 | 0.8831 | 0.7729 | 0.6391 | 0.7475 | 0.9323 |
| both_neighbor_genept | 0.8814 | 0.7699 | 0.6324 | 0.7461 | 0.9311 |
| ensemble | 0.8803 | 0.7694 | 0.6329 | 0.7450 | 0.9304 |
| two_stage_esm2 | 0.8831 | 0.7729 | 0.6391 | 0.7475 | 0.9323 |
| two_stage_genept | 0.8814 | 0.7699 | 0.6324 | 0.7461 | 0.9311 |

**Best**: gene_neighbor_genept (Macro F1: 0.7734)

### REPLOGLE_K562

| Method | Accuracy | Macro F1 | up F1 | down F1 | not_de F1 |
|--------|----------|----------|-------|---------|----------|
| majority_gene | 0.9081 | 0.3331 | 0.0138 | 0.0336 | 0.9518 |
| majority_pert | 0.9080 | 0.3173 | 0.0000 | 0.0000 | 0.9518 |
| pert_neighbor_esm2 | 0.8753 | 0.4716 | 0.2147 | 0.2666 | 0.9335 |
| pert_neighbor_genept | 0.9031 | 0.4151 | 0.1366 | 0.1599 | 0.9488 |
| pert_neighbor_copath | 0.8941 | 0.5531 | 0.3430 | 0.3734 | 0.9429 |
| gene_neighbor_esm2 | 0.9081 | 0.3331 | 0.0138 | 0.0336 | 0.9518 |
| gene_neighbor_genept | 0.9081 | 0.3331 | 0.0138 | 0.0336 | 0.9518 |
| gene_neighbor_copath | 0.9081 | 0.3331 | 0.0138 | 0.0336 | 0.9518 |
| both_neighbor_esm2 | 0.8753 | 0.4716 | 0.2147 | 0.2666 | 0.9335 |
| both_neighbor_genept | 0.9031 | 0.4151 | 0.1366 | 0.1599 | 0.9488 |
| ensemble | 0.8859 | 0.5300 | 0.3054 | 0.3461 | 0.9386 |
| two_stage_esm2 | 0.8611 | 0.4847 | 0.2382 | 0.2906 | 0.9253 |
| two_stage_genept | 0.9036 | 0.4297 | 0.1650 | 0.1750 | 0.9490 |

**Best**: pert_neighbor_copath (Macro F1: 0.5531)

### REPLOGLE_RPE1

| Method | Accuracy | Macro F1 | up F1 | down F1 | not_de F1 |
|--------|----------|----------|-------|---------|----------|
| majority_gene | 0.9055 | 0.5006 | 0.1950 | 0.3572 | 0.9496 |
| majority_pert | 0.9011 | 0.3160 | 0.0000 | 0.0000 | 0.9480 |
| pert_neighbor_esm2 | 0.8772 | 0.5407 | 0.2848 | 0.4041 | 0.9332 |
| pert_neighbor_genept | 0.9021 | 0.5360 | 0.2663 | 0.3943 | 0.9474 |
| pert_neighbor_copath | 0.8914 | 0.5809 | 0.3599 | 0.4420 | 0.9408 |
| gene_neighbor_esm2 | 0.9055 | 0.5006 | 0.1950 | 0.3572 | 0.9496 |
| gene_neighbor_genept | 0.9055 | 0.5006 | 0.1950 | 0.3572 | 0.9496 |
| gene_neighbor_copath | 0.9055 | 0.5006 | 0.1950 | 0.3572 | 0.9496 |
| both_neighbor_esm2 | 0.8772 | 0.5407 | 0.2848 | 0.4041 | 0.9332 |
| both_neighbor_genept | 0.9021 | 0.5360 | 0.2663 | 0.3943 | 0.9474 |
| ensemble | 0.8871 | 0.5795 | 0.3508 | 0.4493 | 0.9384 |
| two_stage_esm2 | 0.8670 | 0.5432 | 0.2933 | 0.4093 | 0.9271 |
| two_stage_genept | 0.9014 | 0.5378 | 0.2698 | 0.3966 | 0.9470 |

**Best**: pert_neighbor_copath (Macro F1: 0.5809)

### HEPG2

| Method | Accuracy | Macro F1 | up F1 | down F1 | not_de F1 |
|--------|----------|----------|-------|---------|----------|
| majority_gene | 0.9601 | 0.3265 | 0.0000 | 0.0000 | 0.9796 |
| majority_pert | 0.9601 | 0.3265 | 0.0000 | 0.0000 | 0.9796 |
| pert_neighbor_esm2 | 0.9408 | 0.4607 | 0.1991 | 0.2134 | 0.9695 |
| pert_neighbor_genept | 0.9564 | 0.4332 | 0.1648 | 0.1571 | 0.9776 |
| pert_neighbor_copath | 0.9548 | 0.5170 | 0.2647 | 0.3096 | 0.9767 |
| gene_neighbor_esm2 | 0.9601 | 0.3265 | 0.0000 | 0.0000 | 0.9796 |
| gene_neighbor_genept | 0.9601 | 0.3265 | 0.0000 | 0.0000 | 0.9796 |
| gene_neighbor_copath | 0.9601 | 0.3265 | 0.0000 | 0.0000 | 0.9796 |
| both_neighbor_esm2 | 0.9408 | 0.4607 | 0.1991 | 0.2134 | 0.9695 |
| both_neighbor_genept | 0.9564 | 0.4332 | 0.1648 | 0.1571 | 0.9776 |
| ensemble | 0.9535 | 0.5139 | 0.2617 | 0.3040 | 0.9761 |
| two_stage_esm2 | 0.9323 | 0.4789 | 0.2207 | 0.2511 | 0.9650 |
| two_stage_genept | 0.9560 | 0.4361 | 0.1690 | 0.1618 | 0.9774 |

**Best**: pert_neighbor_copath (Macro F1: 0.5170)

### JURKAT

| Method | Accuracy | Macro F1 | up F1 | down F1 | not_de F1 |
|--------|----------|----------|-------|---------|----------|
| majority_gene | 0.9742 | 0.3290 | 0.0000 | 0.0000 | 0.9869 |
| majority_pert | 0.9742 | 0.3290 | 0.0000 | 0.0000 | 0.9869 |
| pert_neighbor_esm2 | 0.9558 | 0.3950 | 0.1167 | 0.0908 | 0.9775 |
| pert_neighbor_genept | 0.9669 | 0.3962 | 0.1111 | 0.0943 | 0.9831 |
| pert_neighbor_copath | 0.9637 | 0.4531 | 0.1796 | 0.1980 | 0.9816 |
| gene_neighbor_esm2 | 0.9742 | 0.3290 | 0.0000 | 0.0000 | 0.9869 |
| gene_neighbor_genept | 0.9742 | 0.3290 | 0.0000 | 0.0000 | 0.9869 |
| gene_neighbor_copath | 0.9742 | 0.3290 | 0.0000 | 0.0000 | 0.9869 |
| both_neighbor_esm2 | 0.9558 | 0.3950 | 0.1167 | 0.0908 | 0.9775 |
| both_neighbor_genept | 0.9669 | 0.3962 | 0.1111 | 0.0943 | 0.9831 |
| ensemble | 0.9611 | 0.4262 | 0.1421 | 0.1561 | 0.9802 |
| two_stage_esm2 | 0.9490 | 0.4009 | 0.1249 | 0.1038 | 0.9740 |
| two_stage_genept | 0.9670 | 0.4053 | 0.1142 | 0.1185 | 0.9832 |

**Best**: pert_neighbor_copath (Macro F1: 0.4531)

### TCELL_REST

| Method | Accuracy | Macro F1 | up F1 | down F1 | not_de F1 |
|--------|----------|----------|-------|---------|----------|
| majority_gene | 0.9946 | 0.3324 | 0.0000 | 0.0000 | 0.9973 |
| majority_pert | 0.9946 | 0.3324 | 0.0000 | 0.0000 | 0.9973 |
| pert_neighbor_esm2 | 0.9917 | 0.3344 | 0.0037 | 0.0036 | 0.9958 |
| pert_neighbor_genept | 0.9927 | 0.3347 | 0.0036 | 0.0042 | 0.9963 |
| pert_neighbor_copath | 0.9928 | 0.3532 | 0.0338 | 0.0295 | 0.9964 |
| gene_neighbor_esm2 | 0.9946 | 0.3324 | 0.0000 | 0.0000 | 0.9973 |
| gene_neighbor_genept | 0.9946 | 0.3324 | 0.0000 | 0.0000 | 0.9973 |
| gene_neighbor_copath | 0.9946 | 0.3324 | 0.0000 | 0.0000 | 0.9973 |
| both_neighbor_esm2 | 0.9917 | 0.3344 | 0.0037 | 0.0036 | 0.9958 |
| both_neighbor_genept | 0.9927 | 0.3347 | 0.0036 | 0.0042 | 0.9963 |
| ensemble | 0.9940 | 0.3340 | 0.0026 | 0.0024 | 0.9970 |
| two_stage_esm2 | 0.9879 | 0.3376 | 0.0091 | 0.0097 | 0.9940 |
| two_stage_genept | 0.9924 | 0.3349 | 0.0038 | 0.0048 | 0.9962 |

**Best**: pert_neighbor_copath (Macro F1: 0.3532)

## Best Method per Dataset

| Dataset | Best Method | Macro F1 | up F1 | down F1 |
|---------|-------------|----------|-------|--------|
| adamson | gene_neighbor_genept | 0.7734 | 0.6386 | 0.7487 |
| replogle_k562 | pert_neighbor_copath | 0.5531 | 0.3430 | 0.3734 |
| replogle_rpe1 | pert_neighbor_copath | 0.5809 | 0.3599 | 0.4420 |
| hepg2 | pert_neighbor_copath | 0.5170 | 0.2647 | 0.3096 |
| jurkat | pert_neighbor_copath | 0.4531 | 0.1796 | 0.1980 |
| tcell_rest | pert_neighbor_copath | 0.3532 | 0.0338 | 0.0295 |

## Average Performance Across Datasets

| Method | Avg Macro F1 | Avg up F1 | Avg down F1 | Datasets |
|--------|--------------|-----------|-------------|----------|
| pert_neighbor_copath | 0.5374 | 0.3016 | 0.3492 | 6 |
| ensemble | 0.5255 | 0.2826 | 0.3338 | 6 |
| two_stage_esm2 | 0.5031 | 0.2542 | 0.3020 | 6 |
| pert_neighbor_esm2 | 0.4959 | 0.2430 | 0.2877 | 6 |
| both_neighbor_esm2 | 0.4959 | 0.2430 | 0.2877 | 6 |
| two_stage_genept | 0.4856 | 0.2257 | 0.2671 | 6 |
| pert_neighbor_genept | 0.4808 | 0.2191 | 0.2593 | 6 |
| both_neighbor_genept | 0.4808 | 0.2191 | 0.2593 | 6 |
| gene_neighbor_genept | 0.4325 | 0.1412 | 0.1899 | 6 |
| gene_neighbor_copath | 0.4325 | 0.1412 | 0.1899 | 6 |
| majority_gene | 0.4325 | 0.1412 | 0.1899 | 6 |
| gene_neighbor_esm2 | 0.4325 | 0.1412 | 0.1899 | 6 |
| majority | 0.4325 | 0.1412 | 0.1899 | 6 |
| majority_pert | 0.3191 | 0.0000 | 0.0000 | 6 |
