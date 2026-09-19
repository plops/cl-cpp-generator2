# Tasks: Lambda-`(values ...)`-Fix und Test-Ausbau

Seriell abarbeiten; jeder Schritt endet mit einem Verifikationsgate. Erst
weiter, wenn das Gate gruen ist. Vor jeder `c.lisp`-Aenderung ein Backup
(`cp c.lisp /tmp/c.lisp.known-good`) anlegen.

## 1. PoC-Fix verifizieren (erledigt, 2026-09-19)

- [x] Bug reproduzieren: `(lambda () (declare (values int)) (return 42))`
      emittiert `[&] -> int { ... }`.
- [x] In `parse-lambda` (`c.lisp`) leere Parameterliste als `"()"` emittieren,
      Format auf `"[~{~a~^,~}]~a~@[ -> ~a~] "` stellen.
- [x] Gate: 5 Faelle per `emit-c` geprueft, Ausgabe mit `g++ -std=c++20
      -Wall -Wextra` kompiliert und ausgefuehrt (`42 42 40 5 42`, Exit 0).

## 2. PoC-Suite `t/03_lambda` (erledigt, 2026-09-19)

- [x] `t/03_lambda/lambda-tests.lisp` (6 Faelle: String- + Value-Layer),
      `run.sh`, `README.md` anlegen.
- [x] Gate: `./t/03_lambda/run.sh` → `8 checks, 0 failures`, Exit 0.
- [x] Gate: `./t/02_paren_precedence/run.sh` → `111 checks, 0 failures`
      (keine Regression).
- [x] Gate: Klammer-Balance per Python-Tracker (abzuegl. Strings/Kommentare)
      = 0 in beiden angefassten Dateien. parenmedic meldet False Positives
      (`#\`-Literale, Reader-Conditionals) — SBCL-Load ist massgeblich.

## 3. Commits erstellen (offen)

Conventional Commits nach `plan.md`, Abschnitt 4:

1. `fix(lambda): emit () for parameterless lambdas with return type`
2. `test(lambda): add PoC string and value tests in t/03_lambda`
3. `docs(plan): implementation plan, tasks and walkthrough for lambda tests`

Gate pro Commit: beide Suites gruen, `git status` ohne `build/`-Artefakte.

## 4. Suite ausbauen (offen, je Fall String- + Value-Test)

- [ ] `(capture =)` (by value) und gemischte Captures (`x &y`).
- [ ] Untypisierte Parameter (`auto`-Fallback).
- [ ] Lambda mit Rumpf aus mehreren Formen.
- [ ] Sofort aufgerufene Lambda (`((lambda ...) args)`), falls `emit-c` das
      stuetzt — sonst als Negativ-Test dokumentieren.
- [ ] Klaeren: mehrere `values`-Typen in Lambda → `break` wie `parse-defun`
      oder dokumentieren (vgl. `plan.md`, Punkt 6).
- [ ] Gate: `t/03_lambda/run.sh` und `t/02_paren_precedence/run.sh` gruen.

## 5. Runner zusammenfassen (offen, optional)

- [ ] Skript `t/run_all.sh`, das `t/01_paren` (soweit automatisierbar),
      `t/02_paren_precedence/run.sh` und `t/03_lambda/run.sh` nacheinander
      aufruft und den ersten Fehler propagiert.
- [ ] Gate: ein Durchlauf, Exit 0.

## 6. Doku aus Tests erzeugen (offen, spaeter)

- [ ] Nach dem Vorbild von `cl-rust-generator::generate-documentation` eine
      `SUPPORTED_FORMS.md`-Generierung aus den Testtabellen bauen.
- [ ] Gate: generierte Datei ist aktuell (CI-Check oder Pre-Commit-Hook).
