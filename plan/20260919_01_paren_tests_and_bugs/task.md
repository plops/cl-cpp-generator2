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

## 4. Suite ausbauen (erledigt, 2026-09-19)

- [x] `(capture =)` (by value) und gemischte Captures (`x &y`).
      Befund dabei: `(capture x =)` emittierte `[x,=]` (harter g++-Fehler).
      Fix: `sort-captures` stellt Capture-Defaults (`=`/`&`) nach vorn
      (`[=,x]` kompiliert; `[=,x]` redundant → nur Warnung).
- [x] Untypisierte Parameter (`auto`-Fallback).
- [x] Lambda mit Rumpf aus mehreren Formen.
- [x] Sofort aufgerufene Lambda (`((lambda ...) args)`): funktioniert
      (`(...)(41)`), als `:direct`-Fall im Value-Layer.
- [x] Mehrere `values`-Typen: `break` wie `parse-defun` (statt ungueltigem
      `-> (int, float)`); per Error-Layer getestet (vgl. `plan.md`, Punkt 6).
- [x] Gate: `t/03_lambda/run.sh` → `15 checks, 0 failures`,
      `t/02_paren_precedence/run.sh` → `111 checks, 0 failures`.

## 5. Runner zusammenfassen (erledigt, 2026-09-19)

- [x] Skript `t/run_all.sh` ruft `t/02_paren_precedence/run.sh` und
      `t/03_lambda/run.sh` auf, propagiert den ersten Fehler.
      `t/01_paren` bleibt ausgeschlossen: manueller Workflow mit
      hartcodierten Absolutpfaden, destruktiven `rm`-Schritten und ohne
      Exit-Code-Vertrag (Begruendung im Skriptkopf).
- [x] Gate: ein Durchlauf, Exit 0.

## 6. Doku aus Tests erzeugen (erledigt, 2026-09-19)

- [x] Nach dem Vorbild von `cl-rust-generator::generate-documentation`:
      `t/generate-docs.lisp` + `t/generate_docs.sh [--check]` erzeugen
      `SUPPORTED_FORMS.md` (Repo-Wurzel) aus den Tabellen von `t/02` (79
      Faelle) und `t/03` (12 + 1 Error-Fall). Ausgabe wird live
      re-emittiert, `:description`-Felder (bisher nur `t/03`) als Prosa.
- [x] Gate: `--check` gibt Exit 1 bei veralteter Datei (stale=1, fresh=0
      verifiziert); das Laden der Suites bricht vorher bei roten Tests ab,
      Doku kann nie aus Fehlschlag entstehen.
- [ ] Folgearbeit: `:description`-Felder fuer `t/02`-Faelle nachtragen
      (Generator unterstuetzt sie bereits, gerendert wird auch ohne).
