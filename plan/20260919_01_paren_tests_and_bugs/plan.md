# Implementierungsplan: Lambda mit `(values ...)` und Captures + Unit-Tests

Datum: 2026-09-19.
Auftrag: `plan/20260919_01_paren_tests_and_bugs/prompt.txt`.
Stand: Proof of Concept ist umgesetzt (Fix + `t/03_lambda`), Details in
`walkthrough.md` in diesem Ordner.

## 1. Befund

`parse-lambda` in `c.lisp` (ca. Zeile 663) hat die Parameterliste nur dann
emittiert, wenn Parameter vorhanden waren:

```lisp
(if (null req-param)
    ""
    (funcall emit `(paren ,@...)))
```

mit Format `"[~{~a~^,~}] ~a~@[-> ~a ~]"`. Eine parameterlose Lambda mit
`(declare (values int))` ergab daher `[&] -> int { ... }` — ohne `()`.
`g++ -std=c++20` nimmt das nur als C++23-Extension (`-Wc++23-extensions`),
aeltere Standards und andere Compiler weisen es ab. Ohne Rueckgabetyp
entstand das gueltige `[&] { ... }`.

Fix (2 Zeilen in `parse-lambda`): leere Liste als `"()"` emittieren und das
Format auf `"[~{~a~^,~}]~a~@[ -> ~a~] "` stellen. Ergebnis:

| Eingabe | Ausgabe nach Fix |
|---|---|
| `(lambda () (declare (values int)) ...)` | `[&]() -> int { ... }` |
| `(lambda () ...)` | `[&]() { ... }` |
| `(lambda (a) (declare (type int a) (values int)) ...)` | `[&](int a) -> int { ... }` |
| `(lambda () (declare (capture x) (values int)) ...)` | `[x]() -> int { ... }` |

Keine neuen Abhaengigkeiten, keine API-Aenderung ausser dem korrigierten
String.

## 2. Fehlende Requirements / Vorschlaege

Der Auftrag nennt String-Tests, C++-Programme mit Output-Validierung, ein
Test-Skript und den Blick auf `cl-rust-generator`. Ergaenzend sinnvoll:

1. **Default-Capture dokumentieren/testen.** Leere `capture`-Deklaration
   faellt auf `"&"` zurueck (`parse-lambda`, ca. Zeile 700). Das ist
   beabsichtigt, steht aber nur im Docstring. Ein Value-Test
   (`default-capture-sees-outer-x`) pinnt es fest — im PoC enthalten.
2. **`=` vs `&` Captures.** Explizite `(capture =)` (by value) und gemischte
   Listen (`x &y`) sind emittierbar, aber ungetestet. Naechster Schritt.
3. **Mutable/constexpr/noexcept-Lambdas.** Die DSL kennt dafuer keine
   Deklaration; falls Bedarf besteht, als `(declare (mutable))` o.ae.
   ergänzen — vorerst explizit *nicht* umsetzen, nur vormerken.
4. **String-Normalisierung teilen.** `t/02` vergleicht exakt, `t/03`
   normalisiert Whitespace. Bei weiteren Suites einen gemeinsamen
   Test-Helper (`t/helpers.lisp`) erwägen, statt zu kopieren.
5. **`SUPPORTED_FORMS.md`-Ansatz aus `cl-rust-generator` portieren.**
   Dort erzeugt `generate-documentation` die Doku aus den Testfaellen — jede
   Doku-Zeile ist ein ausgefuehrter Test. Fuer `cl-cpp-generator2` waere das
   der naechste grosse Schritt (siehe Task 6).
6. **Mehrfache `values`-Typen.** `parse-lambda` emittiert bei >1 Typen ein
   `(paren ...)` als Rueckgabetyp — das ist kein gueltiges C++. Entweder
   `break` (wie `parse-defun`) oder dokumentieren. Unabhaengig vom PoC klaeren.

## 3. Kontext-Dateien fuer einen unabhaengigen Agenten

| Datei | Wozu lesen |
|---|---|
| `c.lisp`, `parse-lambda` (ca. 663–729) | Die zu fixende Funktion; `consume-declare` liefert `return-values`/`captures` |
| `c.lisp`, `consume-declare` (ca. 150–260) | Woher `values`/`capture`/`type`-Deklarationen kommen |
| `c.lisp`, `parse-defun` (ca. 394–460) | Referenz: korrekte `values`-Behandlung inkl. Fehler bei Mehrfach-Typen |
| `t/03_lambda/lambda-tests.lisp` | PoC-Suite: Tabellenformat, String- plus Value-Layer |
| `t/03_lambda/run.sh`, `t/03_lambda/README.md` | Runner- und Doku-Konvention |
| `t/02_paren_precedence/paren-tests.lisp` | Vorbild-Suite (4 Layer), `cxx-compiler`-Pattern, `build/`-Konvention |
| `t/02_paren_precedence/README.md` | Doku-Konvention fuer Testverzeichnisse |
| `plan/20260830_01_omit_paren_bug/walkthrough.md` | Wie die Vorgaenger-Suite aufgebaut und begruendet wurde |
| `/workspace/src/cl-rust-generator/transpiler-tests.lisp` | Tabellen-Format (`:name/:description/:lisp/:rust/:tags`), `run-value-tests`, `rustfmt`-Check, `generate-documentation` |
| `/workspace/src/cl-rust-generator/run-tests.sh` | Minimaler Runner zum Kopieren |
| `cl-cpp-generator2.asd` | Abhaengigkeiten (keine neuen einfuehren) |
| `.agents/skills/cl-cpp-generator2/SKILL.md` | DSL-Referenz (vor Transpiler-Arbeit laden) |
| `.agents/AGENTS.md` | Klammer-Debugging (Python-Tracker, parenmedic-Einschraenkungen) |

Nicht verfuegbar in dieser Umgebung: DeepWiki-MCP (`plops/cl-cpp-generator2`).
Ersatz: lokale Quellen oben plus `README.md`. Falls DeepWiki-Abfragen spaeter
moeglich sind, dort `parse-lambda`/`consume-declare`-Doku abgleichen.

## 4. Commit-Konvention

Conventional Commits, eine logische Aenderung pro Commit, mit
ausfuehrlicher Beschreibung (Was/Warum/Wie verifiziert):

```
<typ>(<scope>): <kurze Zusammenfassung>

<Body: Ausgangslage, Ursache, Aenderung>
<Test: welche Suites liefen, Ergebnis>
```

Typen: `fix` (Fehlerkorrektur), `test` (nur Tests), `docs` (nur Doku),
`refactor`, `chore`. Beispiele fuer diese Arbeit:

- `fix(lambda): emit () for parameterless lambdas with return type`
- `test(lambda): add PoC string and value tests in t/03_lambda`
- `docs(plan): implementation plan, tasks and walkthrough for lambda tests`

Jeder Commit muss mit gruenen Suites (`t/03_lambda/run.sh`,
`t/02_paren_precedence/run.sh`) erstellt werden; `git status` vorher auf
kollaterale Aenderungen pruefen (z.B. `build/`-Artefakte — sind per
`**/build/` ignoriert).
