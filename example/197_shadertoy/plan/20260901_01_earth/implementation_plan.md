# Implementation Plan: Animated Earth Globe

## Ziel

`gen4.lisp` erzeugt einen eigenständigen Shadertoy-Shader für einen animierten
Erdglobus. Die Szene wird ohne X11 über EGL ausgeführt und bleibt kompatibel mit
dem bestehenden `headless_gpu_runner`.

## Architektur

1. **Datenbeschaffung**
   - `tools/download_earth_data.sh` lädt Natural-Earth-Landpolygone, NASA Blue
     Marble und NASA City Lights nach `tools/cache/`.
   - `tools/bake_earth_data.py` rastert und komprimiert die Quellen in
     `earth_data.lisp`.
   - `tools/cache/` ist ignoriert und wird nicht committed.

2. **Lisp-Generierung**
   - `gen4.lisp` enthält die Marker, Quaternionen und die GLSL-Emitter.
   - Lisp-seitige Quaternionen-Mathematik erzeugt pro Marker eine Orientierung,
     die den Marker zentriert und Norden nach oben ausrichtet.
   - 19 Generierungs-Tests prüfen Koordinaten-Roundtrip, Einheitsquaternionen,
     Zentrierung, Roll und Quaternion-Komposition.

3. **GLSL-Pässe**
   - `common.glsl`: Quaternionen, `qSlerp`, Marker-Tour und gemeinsame Konstanten.
   - `buf0.glsl`: Kamera-State und einmaliges Entpacken der kompakten Erd-Daten in
     eine RGBA32F-State-Textur.
   - `main_image.glsl`: Textur-Lookups, Planet, Relief, Wasser, Wolken,
     Stadtlichter, Atmosphäre, Sterne, Marker und HUD.

4. **Headless-Ausführung**
   - `headless_gpu_runner.cpp` bekommt `--time`, damit ein beliebiger
     Animationszeitpunkt ohne langes Frame-Stepping gerendert werden kann.
   - `glFinish()` pro Simulationsframe verhindert leere FBOs bei langen Batches.

## Daten- und Größenlimits

- Die vier Tabellen umfassen zusammen 3584 `uint32` (14 KB): 2-Bit-Landmaske,
  4-Bit-Küstenfeld, RGB565-Albedo und 4-Bit-Nachtlichter.
- Große PNG-/GIF-Validierungsdateien werden nicht versioniert. Sie sind in
  `.gitignore` eingetragen und mit `run_earth.sh` bzw. `tools/make_animation.py`
  reproduzierbar.
- Die committeden Quellen sind klein genug für Code-Review; die Cache-Dateien
  und lokalen Renderoutputs bleiben außerhalb des Commits.

## Validierung

- `sbcl --noinform --non-interactive --load gen4.lisp`
- `g++ -O3 headless_gpu_runner.cpp -lEGL -lGL -lpng -o headless_gpu_runner`
- Headless-Screenshots bei Berlin, New York und während der SLERP-Fahrt.
- Benchmark auf 720p, 1080p, 1440p und 4K.
- `git diff --cached --check` sowie Prüfung der größten staged Dateien vor dem
  Commit.
