# Contributing

Follow the setup in [README.md](README.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Workflow

1. Create a personal branch from `staging`: `<name>/<task>`.
2. Make focused changes and run `python -m pytest`.
3. Push your branch and open a PR into `staging`, describing the change and tests.
4. Get a teammate's review and passing CI before merging.
5. Merge ready work from `staging` into `main` through a separate PR.

Add tests as features are implemented. Keep secrets and generated files out of commits.

## Issues

Use issues for tasks, bugs, and questions. For bugs, include reproduction steps
and expected behavior. Apply labels under **Issues → Labels**:

- `bug`: something is broken.
- `enhancement`: a feature or improvement.
- `documentation`: documentation changes.
- `question`: clarification needed.
- `good first issue`: a small starter task.
- `help wanted`: assistance welcome.
