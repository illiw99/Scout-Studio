# Default detection profile

Studio defaults to **Default model · automatic GPU / CPU** and
`scout-reviewed-precision@0.5.0`: YOLO26n1280,2.5 samples/s, adaptive fields,
time-scaled tracking, derived play features/wide temporal context, numeric HGB
play classification and two visual boundary heads. The recipe binds all these
settings; copying a classifier into another pipeline does not reproduce results.
[Current quality/speed](model-status.md) is separate from research candidates.

HGB is not Random Forest. Alternative trees, MLPs, LSTMs, Transformers, GRUs and
temporal heads have already been tested; see [research decisions](research-ledger.md).
Manual profile/model choices remain respected. Automatic managed-default upgrades
preserve explicit selections/rollbacks and reject damaged manifests or concurrent
pointer changes. Installing a model does not rewrite saved fields, reviews or
match configuration.

Standard installers include project-trained decision models and configuration.
Startup prepares those locally in the background; first detection downloads the
pinned official YOLO weights. CPU graphs are exported once into the application
cache and reused. Processing dependencies are installed by the private
[setup wizard](runtime-delivery.md). Weights and licenses are listed in the
[model inventory](model-inventory.md). Opening saved reviews/export needs no
person-detector download.

## Build and launcher details

Use the [current build procedure](development.md#build-standalone-installers-and-releases)
for the bootstrap installers. `reference` and `inference` are explicit developer
build profiles, not the current release-delivery architecture. An optional
`--baseline-pack`/`--baseline-sha256` override must be a verified schema 3
decision-only pack; detector-containing build overrides are rejected. Legacy CPU
packs remain importable at runtime. Source/wheels have an explicit model allowlist.

An explicit FFmpeg can be bundled under `_scout_media`; CUDA video decoding is
separate from CUDA inference. Native smoke tests use isolated model stores and
synthetic media. Historical local launcher deployments do not establish which
binary is installed on another computer.

`start-studio.sh` accepts an explicitly selected standalone executable from the
ignored `.scout-studio-runtime` path file. A missing selected executable reports
an error instead of silently changing environments. Removing that path file
restores source-environment fallback. See [maintenance history](maintenance-history.md)
for earlier local deployments and validation limitations.
