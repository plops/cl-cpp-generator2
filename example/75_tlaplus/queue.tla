-- MODULE queue --

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Producers, Consumers, BufCapacity

ASSUME Assumption ==
       /\ Producers # {}
       /\ Consumers # {}
       /\ Producers \intersect Consumers = {}
       /\ BufCapacity \in (Nat \ {0})


VARIABLES buffer, waitSet

vars == <<buffer, waitSet>>

RunningThreads == (Producers \cup Consumers) \ waitSet

Notify == IF waitSet # {}
          THEN \E x \in waitSet: waitSet' = waitSet \ {x}
	  ELSE UNCHANGED waitSet

Wait(t) == /\ waitSet' = waitSet \cup {t}
           /\ UNCHANGED <<buffer>>


Put(t,d) ==
	 \/ /\ Len(buffer)