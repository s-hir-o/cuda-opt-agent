# Publication notes

The public export preserves the existing linear development history with original author and committer timestamps. Private local configuration, secret-key files, environments, Python caches, and scratch drafts were omitted from all historical snapshots. No original repository files or staged changes were reset. Commit hashes necessarily differ; see `history-map.json`.

New commits capture the current implementation, curated October benchmark evidence, and publication documentation. The original suite evidence is retained without rewriting historical paths. `evidence_sha256.json` from the local run was omitted because it includes artifacts deliberately excluded from publication. The Word report hash can still be checked against `report_validation.json`.

Fresh publication validation: 672 passed, 5 skipped using `python -m pytest -q -m "not gpu" -p no:cacheprovider`. Three dependency deprecation warnings were observed. No new GPU speedup measurements were made.

The MIT license selected by the author in the GitHub repository's initial commit is retained in LICENSE.
