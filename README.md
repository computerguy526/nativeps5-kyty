# nativeps5-kyty

The Kyty GPU back end of [nativeps5](https://github.com/computerguy526/nativeps5): KytyPS5's renderer built as a
native arm64 macOS library (`kyty_gpu.dylib`) that a natively translated PlayStation 5 program loads in place of
its graphics driver.

**Start at the entry repository**: https://github.com/computerguy526/nativeps5 has the build instructions
(`BUILDING.md`), the contribution rules (`CONTRIBUTING.md`, `AGENTS.md`), the pre-commit check and the
KosmicKrisp (Vulkan on Metal) patch series this library is run on. This repository is one of its components and is
normally cloned by its `scripts/fetch-components.sh` as `kyty/`.

## Where this comes from

| Layer | Source | Licence |
|---|---|---|
| KytyPS5 | [Jetsku/KytyPS5](https://github.com/Jetsku/KytyPS5), history up to tag `u59-windows-20261024-int21` (744705ee), unchanged | GPL-2.0; the original [Kyty](https://github.com/InoriRus/Kyty) code is MIT ([LICENSES/Kyty-MIT.txt](LICENSES/Kyty-MIT.txt)) |
| AstroHybrid `kyty-hybrid` series | 36 patches from [pcportastrobot](https://github.com/imnotsharingmyname/pcportastrobot), `Source/kyty-hybrid`, applied in order with their author kept | derivative of KytyPS5: GPL-2.0 |
| nativeps5 | the commits on top: arm64 host port, hosted bridge for the nativeps5 runtime, macOS window and presentation, memory and performance work | GPL-2.0 |

Upstream's own README is kept as [docs/KytyPS5-README.md](docs/KytyPS5-README.md). Upstream's screenshots and its
PGO profiles (`tools/pgo/*.profdata`, recorded while running commercial titles) are not carried in this tree; a
profile recorded on a title is never committed here.

**No game data is here and none is accepted.** Not affiliated with or endorsed by Sony Interactive Entertainment
or any game publisher.

## Build (Apple silicon, macOS 26)

```sh
brew install cmake ninja glslang sdl3 vulkan-headers vulkan-loader
git submodule update --init --recursive --depth 1
cmake -S . -B build-arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DKYTY_BUILD_LAUNCHER=OFF
cmake --build build-arm64 -j 4 --target kyty_gpu        # -> build-arm64/kyty_gpu.dylib
```

The entry repository's `scripts/build.sh kyty` runs exactly this. The library is loaded by the runtime of
[nativeps5-recomp](https://github.com/computerguy526/nativeps5-recomp) with `NPS5_GPU=kyty` and
`NPS5_KYTY_GPU_LIB=<path>/kyty_gpu.dylib`; the interface is `src/hybrid/kytyGpuApi.h`. Run it with the settings in
the entry repository's `titles/kyty-preset.env` (the release preset, `tools/u59-preset.json`): the code defaults
are the slow path. The emulator and launcher targets of upstream are not built or maintained in this fork.

## Licence

GPL-2.0 ([LICENSE](LICENSE)), as upstream. Credits: [NOTICE](NOTICE).
