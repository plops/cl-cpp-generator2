# cl-cpp-generator2 — Abhaengigkeiten (`deps.md`)

Keine neuen Abhaengigkeiten eingefuehrt (Vorgabe: Code und Abhaengigkeiten
klein halten). Stand aus `cl-cpp-generator2.asd`, Versionen per
Quicklisp-Dist im Container (2026-09-19).

| System | Version (Dist) | Herkunft | DeepWiki-Anfrage (spaeter) |
|---|---|---|---|
| `alexandria` | 1.0.1 (`alexandria-20241012-git`) | Quicklisp (u.a. `parse-ordinary-lambda-list`) | org noch zu klaeren — kanonisch GitLab `common-lisp`, GitHub-Mirror unverifiziert |
| `cl-ppcre` | 2.1.2 (`cl-ppcre-20250622-git`) | Quicklisp, Edi Weitz (`edicl`) | `edicl/cl-ppcre` |
| `jonathan` | 0.1 (`jonathan-20200925-git`) | Quicklisp, Rudolph-Miller (`Rudolph-Miller`) | `Rudolph-Miller/jonathan` |

Test-/Build-Werkzeuge (keine Lisp-Abhaengigkeiten, im Ubuntu-Container):

| Werkzeug | Version | Wozu |
|---|---|---|
| `sbcl` | 2.6.0.debian | Test-Runner (`--load ... --quit`) |
| `g++` | 15.2.0 (Ubuntu) | Value-Tests kompilieren (`-std=c++20`), Syntax-Orakel |
| `parenmedic` | `/workspace/src/parenmedic/zig-out/bin/parenmedic` | Klammer-Diagnose (mit False Positives, siehe Walkthrough) |
| `python3` | System | Klammer-Balance-Tracker aus `.agents/AGENTS.md` |

Regel bei kuenftigen Abhaengigkeiten: hier eintragen (System, Version,
GitHub-Organisation), damit DeepWiki-Abfragen konstruierbar bleiben; fuer
Rust-Abhaengigkeiten jeweils auf die neueste vorhandene Version wechseln und
Usage-Beispiele in den Plan aufnehmen (Auftragsvorgabe — diesmal nicht
angewendet, da nichts Neues eingefuehrt wurde).
