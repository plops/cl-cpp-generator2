# Walkthrough: Earth Shader im X11/Vulkan-Launcher

## Ausgangspunkt

Der Earth-Shader wurde bereits von `gen4.lisp` nach
`vulkan-shadertoy-x11/launcher/shaders/earth/` generiert und mit
`headless_gpu_runner` über EGL ausgeführt. Dieser Pfad verwendet zwei
Shadertoy-Pässe:

- `buf0.glsl` schreibt Animationszustand und entpackte Kartendaten in Buffer A.
- `main_image.glsl` liest diesen Buffer über `iChannel0` und zeichnet die Erde.

Für eine interaktive Anzeige steht ein separater Vulkan/XCB-Launcher unter
`vulkan-shadertoy-x11/` zur Verfügung. Er kompiliert feste GLSL-Wrapper nach
SPIR-V, führt vier Offscreen-Buffer aus und bindet den aktuellen Buffer-A-
Output im Main-Pass als `iChannel0`.

## Umsetzung

### Earth-Pässe im Vulkan-Wrapper auswählen

Die Wrapper wurden gezielt auf die Earth-Quellen umgestellt:

- `vulkan-shadertoy-x11/launcher/shaders/src/buf.frag` inkludiert jetzt
  `../earth/common.glsl` und `../earth/buf0.glsl`.
- `vulkan-shadertoy-x11/launcher/shaders/src/main.frag` inkludiert jetzt
  `../earth/common.glsl` und `../earth/main_image.glsl`.

Buffer B, C und D bleiben auf den generischen `shadertoy/`-Quellen. Das ist
notwendig, weil der Vulkan-Launcher immer vier Buffer kompiliert und rendert;
die generischen Folgepässe benötigen Hilfsfunktionen aus dem generischen
`common.glsl`, die der Earth-Shader nicht bereitstellt. Die Earth-Implementierung
benötigt dagegen nur Buffer A und den Main-Pass.

### GLSL-450-Kompatibilität

Der erste Vulkan-Compile schlug fehl, weil der generierte Earth-Shader den
Bezeichner `active` verwendete. Dieser ist in Vulkan GLSL 450 reserviert.
`gen4.lisp` wurde deshalb angepasst, um stattdessen `activeMarker` zu
generieren. Anschließend erzeugt `sbcl --noinform --non-interactive --load
gen4.lisp` Vulkan-kompatible Earth-GLSL-Dateien.

### X11-Startskript

`run_earth_x11.sh` wurde angelegt und ausführbar gemacht. Es:

1. prüft, dass `$DISPLAY` gesetzt ist,
2. generiert den Earth-GLSL aus `gen4.lisp`,
3. kompiliert die Vulkan-Shader-Wrapper mit
   `vulkan-shadertoy-x11/launcher/shaders/build_shaders.sh`,
4. baut und staged den XCB/Vulkan-Launcher mit
   `vulkan-shadertoy-x11/build_scripts/build_linux_x11/build.sh`,
5. wechselt in das Staging-Verzeichnis und startet `./VK_shadertoy`.

Der Verzeichniswechsel in Schritt 5 ist wichtig: Der Launcher lädt
`shaders/spv/*.spv` relativ zum aktuellen Arbeitsverzeichnis. Die erste
Fassung führte `VK_shadertoy` aus dem Projektwurzelverzeichnis aus; dies führte
zu `Could not load the shaders`, weil die SPIR-V-Dateien dort nicht gefunden
wurden. Das Skript startet den Prozess nun aus
`vulkan-shadertoy-x11/build_scripts/build_linux_x11/`, in dem der Build die
`shaders/`-Assets staged.

## Verwendung

```bash
cd /home/kiel/stage/cl-cpp-generator2/example/197_shadertoy

# Interaktives X11-Fenster, Standard: 1280x720
./run_earth_x11.sh

# Eigene Fenstergröße
./run_earth_x11.sh --resX 1920 --resY 1080

# Ein Frame als BMP capturen und schließen
./run_earth_x11.sh --resX 640 --resY 360 --screenshot_and_close
```

Das Skript benötigt ein autorisiertes X11-Display, `sbcl` mit
`cl-cpp-generator2`, `glslangValidator`, `gcc`, XCB- und Vulkan-
Entwicklungsbibliotheken sowie einen Vulkan-Grafiktreiber.

## Validierung

Die Implementierung wurde auf einem X11-System mit einer NVIDIA RTX A4000 und
Vulkan getestet:

- `gen4.lisp` lief erfolgreich; alle **19/19** Earth-Tour-Mathematiktests
  bestanden.
- `build_shaders.sh` kompilierte Main, Buffer A und die generischen Buffer
  B/C/D erfolgreich nach SPIR-V.
- `build.sh` erstellte `VK_shadertoy` und staged die Shader-Assets.
- Ein interaktiver 640×360-Liveness-Test blieb fünf Sekunden aktiv.
- Der Ein-Frame-Test
  `./run_earth_x11.sh --resX 640 --resY 360 --screenshot_and_close`
  initialisierte XCB, verwendete die NVIDIA RTX A4000, meldete
  `screenshot done` und erzeugte
  `vulkan-shadertoy-x11/build_scripts/build_linux_x11/screenshot_0.bmp`.

`--screenshot_and_close` beendet den Launcher absichtlich nach der Aufnahme;
das normale interaktive Startkommando bleibt bis zum Schließen des Fensters
aktiv.
