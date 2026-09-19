# Walkthrough: Lambda-`(values ...)`-Bug + PoC-Tests

Datum: 2026-09-19.
Auftrag: `plan/20260919_01_paren_tests_and_bugs/prompt.txt`.
Arbeit von Wol Pumba beauftragt; Container: Ubuntu mit SBCL 2.6.0, g++ 15.2.

## 1. Was implementiert wurde

**Fix in `c.lisp`, `parse-lambda` (2 Zeilen).** Leere Parameterliste wird als
`"()"` statt `""` emittiert, Format von `"[~{~a~^,~}] ~a~@[-> ~a ~]"` auf
`"[~{~a~^,~}]~a~@[ -> ~a~] "` gestellt. Nebeneffekt (beabsichtigt):
parameterlose Lambdas ohne Rueckgabetyp geben jetzt `[&]() { ... }` statt
`[&] { ... }` — beides gueltig, die neue Form entspricht dem Stil der
generierten Beispiele (`example/131_sdr/...: ([&]() {`).

**PoC-Suite `t/03_lambda/`** nach dem Vorbild von `t/02_paren_precedence`:
`lambda-tests.lisp` (6 Faelle, String- plus Value-Layer), `run.sh`, `README.md`.
Der Value-Layer generiert `build/lambda_value_tests.cpp` (per `**/build/`
ignoriert), kompiliert mit `g++ -std=c++20` und prueft Ruckgabewerte —
damit sind Rueckgabetyp *und* Captures (`x`, `x y`, Default-`&`) semantisch
abgedeckt, nicht nur als String.

**Plan-Dokumente** in `plan/20260919_01_paren_tests_and_bugs/`:
`plan.md` (Befund, 6 Vorschlaege fuer fehlende Requirements, Datei-Guide fuer
einen unabhaengigen Agenten, Commit-Konvention), `task.md` (serielle Schritte
mit Gates), `deps.md` (keine neuen Abhaengigkeiten).

## 2. Verifikation

- `./t/03_lambda/run.sh` → `8 checks, 0 failures`, Exit 0 (6 String-, 1
  Compile-, 1 Run-Check; das C++-Programm meldet `0 value failures`).
- `./t/02_paren_precedence/run.sh` → `111 checks, 0 failures` (keine
  Regression durch den Fix).
- Vor dem Fix per `g++ -std=c++20 -fsyntax-only` bestaetigt: `[&] -> int`
  zieht `-Wc++23-extensions` (ohne C++23 ungueltig), `[&]() -> int` ist
  warnungsfrei.
- Klammer-Balance (Python-Tracker aus `.agents/AGENTS.md`, Strings und
  `;`-Kommentare uebersprungen): 0 in `c.lisp`-Diff-Region und
  `t/03_lambda/lambda-tests.lisp`. Backup vor der Aenderung:
  `/tmp/c.lisp.known-good`.

## 3. Unerwartete Findings

- **parenmedic meldet False Positives.** Im unveraenderten `c.lisp`
  ("9 extra closing parentheses", v.a. um `#-nil`/`#+nil`-Reader-Conditionals)
  und in der neuen Testdatei (`#\Space`-Char-Literale). Massgeblich ist der
  SBCL-Load plus der String-sensible Balance-Tracker — beide gruen.
- **`(sb-ext:quit :code 1)` ist falsch.** Die Option heisst `:unix-status`;
  erste Version der Suite lud mit Style-Warning. Analog pruefen, falls andere
  Skripte `:code` verwenden.
- **`destructuring-bind` mit `&key` ohne `&allow-other-keys`** bricht bei
  unerwarteten Schluesseln hart ab (erster Suite-Lauf). Bei wachsenden
  Tabellen defensiv `&allow-other-keys` erwägen oder alle Schluessel
  destrukturieren.
- **DeepWiki-MCP und Rust-Toolchain-Schritte aus dem Auftrag waren nicht
  anwendbar**: kein MCP in dieser Umgebung (durch lokale Quellen ersetzt),
  keine neue Abhaengigkeit (daher kein Versionswechsel noetig).
- **xvfb nicht benoetigt**: reine Compiler-Tests ohne GUI/GL-Kontext.

## 4. Learnings

- Der Value-Layer ist das eigentliche Orakel: Der String-Test haette auch die
  falsche Erwartung `[&] -> int` "bestaetigen" koennen; erst `g++` beweist die
  Gueltigkeit. String-Tests pinnen, Value-Tests beweisen.
- `t/02` als Schablone (Tabellenformat, `cxx-compiler`-Suche,
  `build/`-Konvention, Skip-ohne-Compiler) macht neue Suites billig — `t/03`
  ist in einem Durchgang entstanden.
- Zwei-Zeilen-Fix, null neue Abhaengigkeiten: Die hacekleine Loesung war hier
  die richtige; kein Umbau von `parse-lambda` noetig.

## 5. Moegliche Erweiterungen (Details in `plan.md`/`task.md`)

`=`-/Misch-Captures, `auto`-Fallback-Parameter, mehrteilige Rümpfe,
sofort aufgerufene Lambdas, Mehrfach-`values` klaeren, `t/run_all.sh`,
`SUPPORTED_FORMS.md`-Generierung nach Rust-Vorbild.

## 6. Programme fuer den Docker-Container

Nichts Neues noetig. Benoetigt und vorhanden: `sbcl` (mit Quicklisp),
`g++`/`clang++`, `python3`, `parenmedic`-Binary. Kein `xvfb`, keine
Rust-Toolchain, keine zusaetzlichen Lisp-Libraries.

## 7. Fortsetzung: Schritte 4–5 (2026-09-19)

Zwei echte Transpiler-Bugs gefunden und behoben (jeweils per `g++`
bestaetigt, vorher/nachher):

- `(capture x =)` → `[x,=]` (Fehler: `expected identifier before '='`).
  Neu: `capture-default-p` + `sort-captures` (`c.lisp`) stellen `=`/`&`
  nach vorn → `[=,x]` kompiliert.
- `(values int float)` in Lambda → `-> (int, float)` (ungueltig). Neu:
  `break` wie `parse-defun` ("multiple return values unsupported").

Suite auf 12 Faelle + 1 Error-Test gewachsen (`15 checks, 0 failures`),
u.a. IIFE (`:direct`-Modus im Value-Layer), `auto`-Parameter,
mehrteilige Ruempfe. Learning: `break` umgeht `handler-case`
(`invoke-debugger`), daher faengt der Error-Layer die Meldung per
werfendem `sb-ext:*invoke-debugger-hook*` ab. `t/run_all.sh` fasst
`t/02` + `t/03` zusammen (Exit 0); `t/01` bleibt wegen hartcodierter
Pfade aussen vor.

## 8. Schritt 6: SUPPORTED_FORMS.md (2026-09-19)

`t/generate-docs.lisp` + `t/generate_docs.sh [--check]` portieren den
Rust-Ansatz (`generate-documentation`): 92 Fall-Abschnitte (79 Klammer-,
12 Lambda-, 1 Error-Fall) mit Lisp-Form, live re-emittiertem C++,
Werten und — soweit in den Tabellen vorhanden — Prosa. Stolpersteine:

- `t/02` und `t/03` definieren je ein inkompatibles `emit-str`; der
  Generator ruft `emit-c` direkt auf statt sie zu teilen.
- `~S` druckte `cl-cpp-generator2::`-Prefixes, bis `*package*` wie im
  Rust-Generator explizit gebunden wurde.
- Das Laden der Suites fuehrt sie aus (Abbruch bei Rot) — die Doku
  entsteht dadurch garantiert nur aus gruenen Tabellen.

`--check`-Gate verifiziert (stale → Exit 1, fresh → Exit 0).

## 9. Folgearbeit: t/02-Descriptions (2026-09-19)

Alle 79 `t/02`-Faelle haben jetzt `:description`-Prosa; vorhandene
Inline-`;;`-Kommentare (mit den kaputten Rechenwegen) sind in die Felder
eingeflossen statt daneben zu stehen. `SUPPORTED_FORMS.md` liest sich
dadurch als echte Doku. Gates: `run_all.sh` Exit 0 (111 + 15 Checks),
`--check` Exit 0, Klammer-Balance neutral (9 = HEAD).

## 10. t/03 vervollstaendigt (2026-09-19)

Der PoC ist zur vollstaendigen Suite gewachsen (21 Faelle + 1 Error-Test,
`24 checks, 0 failures`): String-Default-Reorder (`[&,x]`), Init-Capture
(`[x = 5]`), `this`-Capture und Lambda-als-Argument (beide String-only,
kein Objekt/Callee im Harness), `auto`-Parameter mit `values`,
`void`-Setter, verschachtelter IIFE-Return, `&optional`-Cutoff und
Pointer-Return. Dafuer bekam der Value-Layer `:check`-Faelle (eigene
Bedingung statt `got == value`) und `:void`-Faelle (Aufruf ohne
Rueckgabewert) sowie einen `int c = 0` im Setup; Eintraege ohne `:value`
werden uebersprungen. `noexcept`/`const`/`mutable` bleiben ungetestet,
weil `parse-lambda` sie ignoriert (bewusst, vgl. `plan.md` Punkt 3).
README ohne PoC-Vermerk, Doku-Generator rendert `:check`-Faelle als
"Verified by the custom condition". 101 Fall-Abschnitte in
`SUPPORTED_FORMS.md`.

## 11. Dual-Mode-Tests fuer t/03 (2026-09-19)

Frage aus Review: Warum so viele Klammern — wird Omit nicht mitgetestet?
Antwort: Der Omit-Flag propagiert via `emit`-Closure bis in Lambda-Ruempfe
(`(a)+(b)` → `a+b` verifiziert), aber `t/03` forderte Omit nie an. Jetzt wie
vorgeschlagen und nach Rust-Vorbild dual: String-Layer prueft `:expected`
(full) und `:omit` je Fall (42 Checks), Value-Layer kompiliert zwei
Programme (`_full`/`_omit`), beide muessen identische Werte liefern
(`47 checks, 0 failures`). Doku zeigt beide Varianten. Nebenbefund: `setf`
elidiert ebenfalls (`(c)=(42)` → `c=42`).

## 12. Dual-Abdeckung fuer alle Suites (2026-09-19)

Nachfrage: beide Modi ueberall testen und vergleichen. Audit-Ergebnis:

- `t/03`: bereits dual (String + zwei kompilierte Programme).
- `t/02`: Value-Layer und Random-Test waren dual, aber 75 von 79 Faellen
  hatten kein `:full`. Jetzt pinnt jeder Fall beide Strings
  (`193 checks, 0 failures`); die 8 Faelle ohne `:value` wurden einzeln
  geprueft (Mitgliederzugriff auf Ints, Floats, mutierende Zuweisung und
  undeklarierte Namen laufen im Integer-Harness nicht — Legende in der
  Tabelle).
- `t/01`: 31/37 Faelle existierten in `t/02` bereits; `andeq0`/`singleor0`
  sind Duplikate, `ternary4` ist kein kompilierbares C++. Die restlichen
  drei (`call0`, `insertion0`, `string0`) sind als String-only-Faelle
  portiert (Call-Ketten, Stream-Insertion, String-Literale). `t/01` selbst
  bleibt manueller Workflow (hartcodierte Pfade, destruktive Schritte).
- Fast schiefgegangen: mein Einfuege-Skript trennte beim letzten
  Tabelleneintrag `)))` auf (Entry/Liste/`defparameter`) — per SBCL-Read
  gefunden, Backup eingespielt, Skript korrigiert (alle schliessenden
  Klammern wandern mit). 104 Doku-Abschnitte.
