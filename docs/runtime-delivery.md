# Small installers and automatic setup

Scout Studio installers include the native review interface, media/export support
and approved decision models. The first-launch wizard prepares processing tools
from pinned upstream Python wheels in a private environment. No giant runtime
archives or separate support releases are needed.

A working NVIDIA driver selects CUDA on Windows/Linux; other computers use CPU.
macOS uses its native architecture with Metal when available. Each prepared
engine passes a worker probe, and unavailable CUDA processing falls back to CPU.
Setup never changes a system Python or graphics driver.

Initial setup and repair need an internet connection to download a verified uv
bootstrap, managed Python and hash-pinned dependencies from their upstream
sources. The installer embeds the exact recipes and Scout worker wheel. Progress,
cancellation and retry run outside the Qt UI thread. Subsequent launches reuse
the verified private engine. Skipping setup leaves saved reviews and clip export
available offline. Detection reopens setup when needed.

Processing uses the existing isolated worker job protocol. Footage, fields,
review files, model packs and detection caches retain their existing behavior.
Project decision models remain bundled; generic detector weights come separately
from pinned official sources. Source/Python installations keep their explicitly
installed dependencies.

Each native installed smoke test provisions and probes every platform recipe,
then exercises a worker job, setup wizard, playback/export and uninstall.
Promotion verifies the exact tested bytes and publishes only four installers:
Linux deb, Windows exe and two native macOS dmg files. Build identities, recipe
hashes and installed-test evidence remain in the corresponding Actions artifacts.
GPU-less CI verifies CUDA wheel support and CPU fallback; it does not establish
real NVIDIA hardware performance. See `runtime-delivery-contract.md` for the schema.

## Validation — 13 November 2026

The Linux CPU recipe was installed from upstream into a new private Python and
passed the real worker compatibility probe. Local source checks passed 440 tests
with two platform skips; source and wheel packaging checks passed. Focused
recovery fixtures cover altered shared Python, damaged packages, malformed
receipts, permission changes, cancellation and user-environment isolation.
Native Windows/macOS setup and upgrade checks must pass in Actions before the
1.0.3 tag is promoted. The default upgrade baseline is the published 1.0.2,
with explicit installer SHA-256 pins; repository variables can override it.
