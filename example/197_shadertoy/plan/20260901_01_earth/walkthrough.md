# Walkthrough: Earth Globe Shader

## Ausgangspunkt

`doc/headless.md` beschreibt den NVIDIA-EGL-Workflow im Container. EGL wurde
gegen den NVIDIA-Treiber initialisiert; der Renderer wurde als NVIDIA RTX A4000
mit OpenGL 4.6 erkannt.

## Umsetzung

- `gen4.lisp` wurde als eigener Generator ergänzt.
- Verwendete reale Daten:
  - Natural Earth 110m Landpolygone: Land-/Küsteninformation
  - NASA Blue Marble: Albedo
  - NASA Earth's City Lights: Nachtbeleuchtung
- Die Kamera besucht Berlin, New York, Rio, Kapstadt, Tokio und Sydney.
- Die Marker-Orientierungen werden in Lisp berechnet; im Shader interpoliert
  `qSlerp` zwischen den Orientierungen.
- Wolken, Sterne, Atmosphäre, Stadtlichter, Marker-Ringe, Lichtstrahlen und HUD
  sind animiert.

## Herunterladen und Backen

```bash
cd /workspace/src/cl-cpp-generator2/example/197_shadertoy
sh tools/download_earth_data.sh
python3 tools/bake_earth_data.py
sbcl --noinform --non-interactive --load gen4.lisp
```

`download_earth_data.sh` verwendet `curl`, überspringt vorhandene Dateien und
unterstützt `--force`. `bake_earth_data.py` kann die Dateien ebenfalls beim
Erstlauf nachladen. Beide legen keine Quelldaten im Git-Repository ab.

## Headless-Validierung

```bash
g++ -O3 headless_gpu_runner.cpp -lEGL -lGL -lpng -o headless_gpu_runner
./headless_gpu_runner --res 1920 1080 --frames 2 --time 2 \
  --shader-dir vulkan-shadertoy-x11/launcher/shaders/earth \
  --screenshot /tmp/earth_berlin.png
python3 tools/make_animation.py /tmp/earth_tour.gif 0 8 48 480 300
```

Die Screenshots wurden angeschaut und zeigen die Erde mit Küsten, Terrain,
Wolken, Atmosphäre und Marker-Ringen. Eine Kontaktansicht deckt die sechs
Marker ab; eine separate Falschfarbenansicht prüfte Wolken, Landmaske,
Albedo, Beleuchtung und Markerpfade.

## Messwerte

Auf der RTX A4000 mit Treiber 610.57.04 wurden zuletzt gemessen:

| Auflösung | FPS | mittlere Framezeit |
|---|---:|---:|
| 1280×720 | 1038 | 0.96 ms |
| 1920×1080 | 494 | 2.02 ms |
| 2560×1440 | 286 | 3.50 ms |
| 3840×2160 | 129 | 7.74 ms |

Die großen lokalen Renderoutputs (PNG/GIF) werden absichtlich nicht committed;
`.gitignore` schützt vor einem versehentlichen Hinzufügen.
