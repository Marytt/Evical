# Data access

No benchmark question text is distributed in this repository.

- **MedAbstain (MedQA-derived NA/NAP):** fetched at run time by src/data/dataset.py from
  the MedAbstain GitHub repository (file names in configs/pipeline_config.yaml).
  Review the MedAbstain and MedQA terms before use.
- **MediQ:** download `data/all_dev_good.jsonl` from the stellalisy/mediQ repository after
  reviewing its terms, then run scripts/prepare_mediq.py (expected source SHA-256 in its docstring).
- **Length-matched control:** built locally by scripts/build_length_control.py from MedAbstain.
