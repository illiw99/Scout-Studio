# Contributing

See the [development guide](docs/development.md) for environment setup, tests,
package checks, and commit conventions. Use the [code map](docs/code-map.md) to
find the relevant source files.

Scout Studio's own code and documentation are AI-generated under the maintainer's
guidance; see the [project background](README.md#ai-generated-human-guided).
Changes should make their intent clear and include relevant validation. AI
generation does not establish that a change is correct or ready to release.

Keep changes focused and validate them with temporary fixtures. Preserve user
footage, field polygons, detection caches, trained models, saved reviews, and
exports. Native playback changes need a native smoke test as well as offscreen
checks.

When using agents, follow [AGENTS.md](AGENTS.md). Commit completed changes
regularly, without coauthor trailers or assistant attribution. Publishing and
pushing require explicit authorization.
