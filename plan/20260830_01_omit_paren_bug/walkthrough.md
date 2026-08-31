# Klammer-Elision in `emit-c` (`:omit-parens t`) — Untersuchung und Korrektur

Datum: 2026-08-30
Auftrag: `plan/20260830_01_omit_paren_bug/prompt.txt`

## 1. Ausgangslage

Beim Bau von `example/197_shadertoy` (Earth-Shader) fiel auf, dass ein negierter
Ausdruck vor einer Division seine Klammern verliert:

```lisp
(exp (/ (- (- perp EARTH_R)) 0.055f0))
```

| Modus | Ausgabe vor dem Fix |
| --- | --- |
| voll geklammert | `exp(( -((perp)-(EARTH_R)))/(5.50e-2F))` |
| `:omit-parens t` | `exp( -perp-EARTH_R/5.50e-2F)` ← **falsch** |

Im Shader äußerte sich das als verschwundener Atmosphären-Halo
(`g ≈ e^-19`, dunkler Ring am Rand). Der Workaround im Beispiel war eine
Zwischenvariable — die Ursache lag aber im Generator.

## 2. Wo die Entscheidung fällt

`c.lisp` enthält zwei Tabellen und eine Entscheidungsstelle:

* `*operators*` (Zeile ~792) — flache Liste „bekannter“ Formen.
* `*precedence*` (Zeile ~864) — die eigentliche C++-Präzedenztabelle
  (Zeilenindex = Bindungsstärke, kleiner Index = bindet stärker) mit
  Assoziativität.
* Die Form `paren*` in `emit-c` — sie bekommt den *Elternoperator* und ein
  Argument und entscheidet, ob das Argument geklammert werden muss. Jeder
  Operator-Handler (`+`, `-`, `*`, `/`, `==`, …) ruft `paren*` für seine
  Operanden auf.

Bei `:omit-parens nil` klammert `paren*` bedingungslos alles — dieser Modus ist
das Referenz-Orakel für die Tests.

## 3. Befunde

### 3.1 Hauptursache: `-unary` fehlte in `*operators*`

Der `-`-Handler unterscheidet unäres und binäres Minus und benutzt für das
unäre Minus den intern erzeugten Operator `-unary`:

```lisp
(m '-unary (format nil " -~a" (emit `(paren* -unary ,(car args)))))
```

`-unary` steht in `*precedence*` (Zeile der unären Operatoren, Index 5), war
aber **nicht** in `*operators*`. Die Entscheidung in `paren*` lautete jedoch

```lisp
(if (and (member op0 *operators*) (member (car arg) *operators*)) …)
```

Damit fiel jeder Operand eines unären Minus in den `else`-Zweig, der als
„Funktionsaufruf“ interpretiert wird und **grundsätzlich keine Klammern**
setzt. Aus `(- (- perp EARTH_R))` wurde ` -perp-EARTH_R`.

Betroffen war jedes unäre Minus mit zusammengesetztem Operanden:
`(- (- a b))`, `(- (+ a b))`, `(- (aref …))` in Divisions-, Multiplikations-,
Vergleichs-, Ternär- und Zuweisungskontexten.

### 3.2 Zwei Listen, eine Wahrheit

`*operators*` und `*precedence*` waren unabhängig gepflegt und
auseinandergelaufen. Fehlend bzw. falsch:

* `-unary` fehlte in `*operators*` (siehe 3.1),
* `<=>` fehlte in `*operators*`,
* `scope` fehlte in `*operators*`,
* `^=` war in `*precedence*` als `^-` **vertippt**.

Der Tippfehler führte zu einem harten Absturz, nicht nur zu falschen Klammern:
`^=` ist in `*operators*`, also lief die Präzedenzabfrage los, `lookup-precedence`
lieferte `NIL`, und `(< NIL 7)` warf einen `type-error`:

```
(^= a (+ b c))  ->  SB-KERNEL:TWO-ARG-< NIL 7
```

Konsequenz: Die Entscheidung basiert jetzt ausschließlich auf
`lookup-precedence` (also `*precedence*`). Kein Eintrag = primärer Ausdruck /
Funktionsaufruf = keine Klammern. Damit ist ein fehlender Tabelleneintrag nicht
mehr stumm gefährlich, sondern höchstens konservativ.

### 3.3 Die Abkürzung „zwei Elemente brauchen keine Klammern“

```lisp
((<= (length arg) 2)
 ;; two or one elements doesn't need paren
 …)
```

Diese Abkürzung war für Funktionsaufrufe (`(H x)`) richtig, für unäre
Operatoren aber falsch: `(- (- a b))` ist eine zweielementige Liste, emittiert
aber `-a-b` und gruppiert damit um. Die Abkürzung ist entfernt; die
Präzedenzprüfung behandelt jetzt Listen jeder Länge.

Damit sie das kann, brauchte es eine Übersetzung von „Form“ auf „tatsächlich
emittierter Operator“ (neu: `effective-operator`):

| Form | emittiert | effektiver Operator |
| --- | --- | --- |
| `(- x y)` | `x-y` | `-` (Index 7) |
| `(- x)` | `-x` | `-unary` (Index 5) |
| `(+ x)`, `(* x)`, `(or x)` … | nur `x` | der Operator von `x` (rekursiv) |
| `(paren …)`, `(curly …)`, `(& …)` | bringt eigene Klammern | `NIL` (primär) |
| `(< a b c)` | `a<b && b<c` | `logand` |
| `(foo x y)` | `foo(x, y)` | `NIL` (primär) |

Das `(or 255)`-Verhalten war der Grund, warum die Abkürzung überhaupt
existierte: `(bitwise-not (or 255))` muss `~255` liefern und nicht `~(255)`.
Die Rekursion in `effective-operator` erhält dieses Verhalten.

### 3.4 Gleiche Präzedenz wurde nie geklammert (vom Randomtest gefunden)

Die Assoziativitätsklausel war toter Code:

```lisp
(and (eq p0 p1) (not (eq p0assoc p1assoc)))
```

`p0` und `p1` sind Zeilenindizes derselben Tabelle. Gleicher Index bedeutet
gleiche Zeile und damit zwangsläufig gleiche Assoziativität — die Bedingung war
immer `NIL`. Für `-`, `/` und `%` verdeckte ein pauschaler Hammer das Problem
(`(member op0 '(/ % -))`), für alles andere nicht. Der randomisierte
Differenztest fand deshalb:

```lisp
(? (? 1 3 7) x y)   ->  1 ? 3 : 7 ? x : y      ; regruppiert, ?: ist rechtsassoziativ
(< 7 (< 1 7))       ->  7<1<7                  ; regruppiert, < ist linksassoziativ
(== a (== b c))     ->  a==b==c                ; dito
(<< a (<< b c))     ->  a<<b<<c                ; dito
```

`paren*` nimmt jetzt optional die **Position** des Operanden (`l` = linker/erster
Operand, `r` = rechter Operand, `NIL` = egal). Bei gleicher Präzedenz wird
geklammert, wenn der Operand auf der Seite steht, die die Assoziativität nicht
bevorzugt:

* linksassoziativ (`- / % << >> < == …`): rechter Operand → Klammern
* rechtsassoziativ (`?: = += …`): linker Operand → Klammern
* Ausnahme: beide Operatoren sind untereinander assoziativ
  (`+ * ^ & | && ||`), dann bleibt es flach. Deshalb liefert `(+ a (+ b c))`
  weiterhin `a+b+c`, `(* a (/ b c))` aber `a*(b/c)` — `*` und `/` teilen sich
  eine Präzedenzzeile, sind aber nicht untereinander assoziativ.

### 3.5 Fehler, die in **beiden** Modi falschen Code erzeugten

Diese hätte ein reiner „voll geklammert gegen sparsam geklammert“-Vergleich nie
gefunden, weil beide Seiten gleich falsch waren:

* **`cast`** klammerte seinen Operanden nie:
  `(cast int (+ a b))` → `(int) a+b`, also `((int)a)+b` statt `(int)(a+b)`.
* **`dot`** benutzte `emit` direkt statt `paren*`:
  `(dot (- (- a b)) c)` → `-((a)-(b)).c`, also `-(((a)-(b)).c)`.

Beide benutzen nun die neue Hilfsfunktion `binds-looser-p`, die *nur dann*
klammert, wenn es nötig ist — bewusst nicht `paren*`, weil `paren*` im
Standardmodus alles klammert und `(dot obj (method x))` dann zu
`(obj).(method(x))` geworden wäre. Das hätte die Ausgabe fast aller Beispiele
kosmetisch umgeworfen.

### 3.6 `paren*`-Aufrufe mit fehlendem Elternoperator

Drei Stellen riefen `paren*` mit nur einem Argument auf, was direkt in
`(break "paren* expects only two arguments")` lief:

* `(/ x)` (einargumentige Division, emittiert `1.0/x`),
* die dreiargumentige Kettenvergleichs-Variante von `<=`.

Beide sind korrigiert (`(/ a)` → `1.0/a`, `(<= c b a)` → `c<=b && b<=a`).

## 4. Änderungen in `c.lisp`

| Stelle | Änderung |
| --- | --- |
| `*operators*` | `-unary`, `<=>`, `scope` ergänzt; Docstring erklärt `-unary` |
| `*precedence*` | Tippfehler `^-` → `^=` |
| neu `*chain-operators*` | Operatoren, die ihre Argumente verketten |
| neu `*self-delimited-operators*` | Formen mit eigenen Klammern |
| neu `*associative-operators*` | wo gleiche Präzedenz flach bleiben darf |
| neu `effective-operator` | Form → tatsächlich emittierter Operator |
| neu `binds-looser-p` | „muss geklammert werden?“ für `cast` und `dot` |
| `paren*` | Länge-≤2-Abkürzung entfernt, Entscheidung über `lookup-precedence`, optionales Positionsargument, positionsabhängige Assoziativitätsregel |
| `-`, `/`, `%`, `<<`, `>>`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `<=>`, `=`, `/=`, `*=`, `^=`, `incf`, `decf`, `?` | übergeben jetzt `l`/`r` als Operandenposition |
| `/` | `(/ x)` übergibt den Elternoperator |
| `<=` | dreiargumentige Variante übergibt Elternoperator und Position |
| `cast` | klammert den Operanden, wenn er lockerer bindet |
| `dot` | dito |

## 5. Ergebnis

```lisp
(exp (/ (- (- perp EARTH_R)) 0.055f0))
;; -> exp(( -(perp-EARTH_R))/5.50e-2F)     korrekt
```

## 6. Tests: `t/02_paren_precedence/`

`./t/02_paren_precedence/run.sh` — 111 Checks, Exit-Code 1 bei Fehlern.
Vier Schichten, absichtlich mit unterschiedlichen Fehlermodi:

1. **String-Tests** — Vergleich gegen handverifizierte Referenz-Strings, für
   den sparsamen *und* (wo eingetragen) den voll geklammerten Modus. Nur diese
   Schicht findet Fehler, die in beiden Modi gleich falsch sind (`cast`, `dot`).
   Enthält alle Referenzen aus `t/01_paren/gen00.lisp` als Regressionsnetz.
2. **Wert-Tests** — ein generiertes C++-Programm rechnet jeden Ausdruck in
   beiden Modi aus und vergleicht mit einem erwarteten Integer. Semantische
   Absicherung: eine fehlende Klammer ändert den Wert.
3. **Helper-Tests** — Unit-Tests für `effective-operator` und `binds-looser-p`.
4. **Randomisierter Differenztest** — 400 zufällige Ausdrücke (Tiefe 3, fester
   Seed) werden in beiden Modi emittiert, in *einem* C++-Programm ausgewertet
   und paarweise verglichen. Der voll geklammerte Modus ist das Orakel. Nur
   UB-freie Formen werden erzeugt (Divisoren sind positive Literale,
   Shift-Operanden werden maskiert).

Schicht 4 hat die Assoziativitätsfehler aus 3.4 gefunden, die in den
handgeschriebenen Fällen nicht vorkamen. Zur Absicherung wurden zusätzlich
manuell 8 × 2000 Ausdrücke mit Tiefe 4 und wechselnden Seeds gefahren — ohne
Abweichung.

Gegenprobe: mit dem alten `c.lisp` (via `git stash`) schlagen 17 String-Tests
fehl und der Randomtest meldet Abweichungen — die Tests fangen den Fehler also
wirklich.

## 7. Auswirkung auf bereits generierten Code

`example/197_shadertoy/gen4.lisp` neu ausgeführt, Diff der GLSL-Ausgabe:

* `exp(-e1 * e1)` → `exp(-(e1 * e1))` (redundant, aber die konservativ korrekte
  Form; unäres Minus über `*` ist rechnerisch identisch)
* `-bb - sqrt(d2)` → `(-bb) - sqrt(d2)` (redundant, korrekt)
* `uint(((idx & 15)) * 2)` → `uint((idx & 15) * 2)` (doppelte Klammer entfällt,
  weil `(& …)` jetzt als primär erkannt wird)

Alle Änderungen sind semantikerhaltend; die 19 Selbsttests in `gen4.lisp` laufen
weiter durch.

## 8. Unerwartete Funde am Rand

* **Der Hammer bleibt.** `(member op0 '(/ % -))` / `(member op1 '(/ % -))`
  klammert pauschal, sobald `-`, `/` oder `%` beteiligt sind. Mit der neuen
  Positionsregel wäre `(member op0 …)` streng genommen redundant
  (`(- (+ a b) c)` → `a+b-c` wäre korrekt), aber `t/01_paren` dokumentiert
  ausdrücklich `(3+4)-(7-3)` und `(17/5)+3`. Der Hammer bleibt daher stehen:
  Er kostet nur Lesbarkeit, nicht Korrektheit, und vermeidet Churn in ~195
  Beispielen. Das ist der Grund für die redundanten Klammern in Abschnitt 7.
  `(member op1 '(/ % -))` ist dagegen wirklich nötig: `(* a (/ b c))` darf nicht
  zu `a*b/c` werden (Integer-Division).
* **`+`/`*` werden als assoziativ behandelt**, obwohl das für Gleitkomma nur
  bis auf Rundung gilt. Das war schon vorher so (`t/01_paren` erwartet
  `(7-3)+3+4`) und wurde bewusst nicht geändert, aber in
  `*associative-operators*` dokumentiert.
* **Stale Referenz in `t/01_paren/gen00.lisp`**: Der erwartete String für
  `insertion0` enthält `3.141590f`, der Generator schreibt `3.141590F`
  (Großbuchstabe). Unabhängig von diesem Bug, nicht angefasst.
* **ASDF-Fasl-Falle**: `git stash push c.lisp` / `git stash pop` setzt die mtime
  neu, aber ASDF vergleicht nur sekundengenau. Nach einem Stash-Zyklus innerhalb
  derselben Sekunde lädt SBCL das *alte* `c.fasl` weiter. Wer beim Debuggen
  stasht, sollte
  `rm ~/.cache/common-lisp/sbcl-*/…/cl-cpp-generator2/c.fasl` einplanen —
  sonst testet man minutenlang gegen den falschen Compiler.
* **`(& a b)` erzeugt immer eigene Klammern** (`(a&b)`), `(and a b)` dagegen
  nicht (`a & b`). Beide bedeuten bitweises Und. Diese Asymmetrie ist alt und
  überraschend; sie ist jetzt in `*self-delimited-operators*` explizit
  berücksichtigt, statt implizit über die Längenabkürzung.
* **`logand` ist `&&` und `and` ist `&`.** Der Kommentar in `*precedence*`
  bezweifelt das selbst („I'm never sure … I think it currently is wrong“). Die
  Präzedenzzeilen passen aber zu dieser Belegung, und die Tests (`logorand0`,
  `andeq0`) bestätigen sie. Nicht angefasst.

## 9. Reproduktion

```sh
# Tests
./t/02_paren_precedence/run.sh

# härterer Randomlauf
sbcl --noinform --disable-debugger \
     --load t/02_paren_precedence/paren-tests.lisp \
     --eval '(cl-cpp-generator2::run-random-tests :count 2000 :depth 4 :seed 42)' \
     --quit
```
