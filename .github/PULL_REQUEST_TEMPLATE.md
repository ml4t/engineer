## Owning issue

Closes #

The issue and this PR each describe one problem or coherent change. Separate unrelated fixes.
Automated dependency updates and administrative metadata changes are exempt from the issue
requirement.

## Outcome

Describe the problem, expected behavior, and what this change does.

## Compatibility

- [ ] No public compatibility impact
- [ ] Compatibility impact is documented in the owning issue and migration guidance

## Verification

For a bug fix, record the regression command and its result on both revisions:

- Base revision, command, and failure:
- PR revision, same command, and passing result:

For other changes, state the relevant commands and results.

- [ ] Numerical changes state the intended definition and use independent correctness evidence
- [ ] Verification exercises the affected public API or user workflow
- [ ] Ruff lint and format checks pass
- [ ] `ty` passes
- [ ] Test suite passes
- [ ] Package build passes when applicable
- [ ] Strict MkDocs build passes when documentation is affected
- [ ] Ecosystem qualification passes

## AI assistance

State whether you used AI tools and which parts they helped produce, including code and prose.

- [ ] I reviewed and tested this submission and can explain its code and evidence

## Documentation and release

State the documentation, release notes, and patch-release work required after merge.
