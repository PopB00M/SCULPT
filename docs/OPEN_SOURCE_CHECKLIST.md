# Open Source Checklist

Use this checklist before uploading SCULPT to Anonymous GitHub.

- No real API keys, tokens, passwords, or private URLs are committed.
- `.env` is ignored and only `.env.example` is published.
- README commands run from the repository root.
- Full dataset redistribution has been checked against the dataset license.
- Cached graph artifacts are checked before sharing because they may contain raw code.
- Results and logs do not reveal author names, institutions, user names, or private filesystem paths.
- Package versions, CUDA version, GPU type, and random seeds are recorded for reported experiments.
- Code license and dataset/model notices are included.
