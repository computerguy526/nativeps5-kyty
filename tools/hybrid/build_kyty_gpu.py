"""Build kyty_gpu.dll from an existing kyty_emulator build without rebuilding it.

Reads the kyty_emulator compile flags, object list and link libraries from <kyty build>/build.ninja,
compiles only src/hybrid/kytyGpuBridge.cpp with those flags, and links the emulator's objects
(minus main.cpp.obj) with it into <out>/kyty_gpu.dll using src/hybrid/kyty_gpu.def. The emulator
build directory is only read. Runs inside the Visual Studio x64 environment (vcvars64.bat).

Usage: python tools/hybrid/build_kyty_gpu.py <kyty build dir> <out dir>
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
# Kyty core sources the hybrid branch changes: compiled here with kyty_emulator's flags, their
# objects replace the emulator's in the link (the emulator build directory stays as it is).
HYBRID_SOURCES = (
    "src/kernel/memory.cpp",
    "src/libs/dualSenseBluetooth.cpp",
    "src/libs/dualSenseHaptics.cpp",
)
LLVM_DEFAULT = pathlib.Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin")


def cache_value(build, name):
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(name + ":"):
            return line.split("=", 1)[1]
    return None


def vs_environment():
    output = subprocess.run(f'cmd /s /c ""{VCVARS}" >nul && set"', capture_output=True, text=True, shell=True, check=True).stdout
    env = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if "INCLUDE" not in env or "LIB" not in env:
        sys.exit("vcvars64.bat did not provide INCLUDE and LIB")
    return env


def ninja_block(text, header):
    start = text.find(header)
    if start < 0:
        sys.exit(f"build.ninja has no '{header}'")
    end = text.find("\n\n", start)
    return text[start:end]


def variables(block):
    values = {}
    for line in block.splitlines()[1:]:
        match = re.match(r"\s+(\w+) = (.*)", line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def ninja_paths(text):
    return [part.replace("$:", ":").replace("$ ", " ") for part in re.split(r"(?<!\$) ", text) if part]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build = pathlib.Path(sys.argv[1]).resolve()
    out = pathlib.Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ninja = (build / "build.ninja").read_text(encoding="utf-8", errors="replace")

    compile_block = variables(ninja_block(ninja, "build CMakeFiles\\kyty_emulator.dir\\src\\libs\\agc.cpp.obj:"))
    link_header = ninja_block(ninja, "build kyty_emulator.exe:")
    link_vars = variables(link_header)
    inputs = link_header.splitlines()[0].split(":", 1)[1].split("|")[0]
    objects = [p for p in ninja_paths(inputs)[1:] if p.endswith(".obj") and not p.endswith("main.cpp.obj")]
    libraries = ninja_paths(link_vars["LINK_LIBRARIES"])

    compiler = cache_value(build, "CMAKE_CXX_COMPILER")
    llvm = pathlib.Path(compiler).parent if compiler else LLVM_DEFAULT
    source_root = pathlib.Path(cache_value(build, "CMAKE_HOME_DIRECTORY") or ROOT)
    print(f"toolchain {llvm}, sources {source_root}", flush=True)
    env = vs_environment()
    bridge_obj = out / "kytyGpuBridge.obj"
    compile_cmd = [str(llvm / "clang-cl.exe"), "/nologo", "/c"] + compile_block["DEFINES"].split() + compile_block["FLAGS"].split()
    compile_cmd += [f"/DKYTY_GPU_BUILD_ID={compile_block.get('CONFIG', 'Release')}"]
    for quoted, plain, system in re.findall(r'-I"([^"]+)"|-I(\S+)|-imsvc(\S+)', compile_block["INCLUDES"]):
        if system:
            compile_cmd.append("/imsvc" + system.replace("$:", ":"))
        else:
            compile_cmd.append("/I" + (quoted or plain).replace("$:", ":"))
    print("compile: src/hybrid/kytyGpuBridge.cpp", flush=True)
    subprocess.run(compile_cmd + [f"/Fo{bridge_obj}", str(source_root / "src/hybrid/kytyGpuBridge.cpp")], cwd=build, env=env, check=True)
    for source in HYBRID_SOURCES:
        replaced = "CMakeFiles\\kyty_emulator.dir\\" + source.replace("/", "\\") + ".obj"
        if replaced not in objects:
            sys.exit(f"{source}: no kyty_emulator object to replace")
        own = out / (pathlib.Path(source).name + ".hybrid.obj")
        print(f"compile: {source} (hybrid change, replaces {replaced})", flush=True)
        subprocess.run(compile_cmd + [f"/Fo{own}", str(source_root / source)], cwd=build, env=env, check=True)
        objects[objects.index(replaced)] = str(own)

    response = out / "kyty_gpu.link.rsp"
    lines = ["/NOLOGO", "/DLL", "/MACHINE:X64", "/INCREMENTAL:NO", "/DEBUG:FULL", "/OPT:REF", "/OPT:ICF",
             f"/OUT:{out / 'kyty_gpu.dll'}", f"/IMPLIB:{out / 'kyty_gpu.lib'}", f"/PDB:{out / 'kyty_gpu.pdb'}",
             f"/DEF:{ROOT / 'src/hybrid/kyty_gpu.def'}", f'"{bridge_obj}"']
    lines += [f'"{build / o}"' for o in objects]
    for library in libraries:
        path = pathlib.Path(library)
        lines.append(f'"{path if path.is_absolute() or len(path.parts) == 1 else build / path}"')
    response.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"link: {len(objects)} emulator objects + bridge, {len(libraries)} libraries", flush=True)
    subprocess.run([str(llvm / "lld-link.exe"), f"@{response}"], cwd=build, env=env, check=True)
    winpthread = ROOT / "3rdparty/winpthread/bin/libwinpthread-1.dll"
    target = out / "libwinpthread-1.dll"
    if winpthread.exists() and not target.exists():
        target.write_bytes(winpthread.read_bytes())
    print(f"built {out / 'kyty_gpu.dll'}")


if __name__ == "__main__":
    main()
