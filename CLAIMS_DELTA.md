# Claims delta after the FL warm-start fix rerun

Template. Fill in after the Kaggle rerun of the patched GROUP-A notebook and
the scripted table regeneration (`scripts/regen_table6.py` /
`regen_tables.py`). Every paper claim that depends on FL ordering is listed;
mark each **KEPT** or **CHANGED** with the old and new numbers. Changed claims
are flagged for Chloe — never silently edited in the paper.

Status legend: `PENDING` (rerun not done) / `KEPT` / `CHANGED`.

## Claims to check explicitly

| # | Claim (paper location) | Old evidence | New evidence | Status |
|---|---|---|---|---|
| 1 | FedProx tenfold recovery on SAML at alpha=0.05 | _fill from paper_ | _from rerun CSVs_ | PENDING |
| 2 | PersFL sole survivor on IBM HI-Large | _fill from paper_ | _from rerun CSVs_ | PENDING |
| 3 | FedNova F1 collapse at the operating point | _fill from paper_ | _from rerun CSVs_ | PENDING |
| 4 | MoE Static vs centralised CatBoost on ULB | _fill from paper_ | _from rerun CSVs_ | PENDING |

## Other FL-ordering-dependent claims

Enumerate any additional statements in the paper that rank or compare FedAvg /
FedProx / FedNova / PersFL (or MoE variants built on them) and add a row per
claim. Known context: on the buggy code FedAvg/FedProx/FedNova were arithmetic
variants of the same fixed local models, so **any** ordering among them in the
old tables is suspect until re-derived from the rerun.

| # | Claim (paper location) | Old evidence | New evidence | Status |
|---|---|---|---|---|
| 5 | _..._ | | | PENDING |

## Notes

- Table 6 (per-seed AUPRC, ULB alpha=0.5) is replaced wholesale by
  `scripts/regen_table6.py` output; the FedAvg fingerprint of the old table
  was 0.699 +/- 0.116 (population std).
- Table V (60-round privacy grid, `scripts/v1`/`v2`) is unaffected by the
  warm-start bug and is out of scope here.
