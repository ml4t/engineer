# Contributing

Follow the [ML4T contribution policy](https://github.com/ml4t/ecosystem/blob/main/CONTRIBUTING.md)
for issue scope, pull request scope, regression evidence, compatibility decisions, and AI
disclosure. Submit changes to this repository only when they affect `ml4t-engineer`.

Use the issue forms to describe one problem or coherent change. Bug reports must include the
affected version, environment, expected and observed behavior, and a complete reproduction.
Report suspected vulnerabilities through the private process in [SECURITY.md](SECURITY.md).

Before requesting review, run the repository-specific environment and quality commands in
[README.md](README.md#development). Include the commands and results in the pull request. Bug fixes
must record the same public-workflow regression failing on the relevant base revision and passing
on the proposed revision. Changes to numerical results must state the intended definition and use
independent evidence; preserving existing output does not establish correctness. Documentation
changes must pass the strict MkDocs build.
