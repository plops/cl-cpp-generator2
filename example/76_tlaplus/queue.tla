--------------------------- MODULE queue -----------------------------------

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Producers, Consumers, BufCapacity

ASSUME Assumption ==
       /\ Producers # {}
       /\ Consumers # {}
       /\ Producers \intersect Consumers = {}
       /\ BufCapacity \in (Nat \ {0})

-----------------------------------------------------------------------------
VARIABLES buffer, waitSet
vars == <<buffer, waitSet>>

RunningThreads == (Producers \cup Consumers) \ waitSet

Notify == IF waitSet # {}
          THEN \E x \in waitSet: waitSet' = waitSet \ {x}
	  ELSE UNCHANGED waitSet

Wait(t) == /\ waitSet' = waitSet \cup {t}
           /\ UNCHANGED <<buffer>>

-----------------------------------------------------------------------------
Put(t,d) ==
	 \/ /\ Len(buffer) < BufCapacity
	    /\ buffer' = Append(buffer, d)
	    /\ Notify
	 \/ /\ Len(buffer) = BufCapacity
	    /\ Wait(t)

Get(t) ==
       \/ /\ buffer # <<>>
          /\ buffer' = Tail(buffer)
	  /\ Notify
       \/ /\ buffer = <<>>
          /\ Wait(t)

-----------------------------------------------------------------------------
Init == /\ buffer = <<>>
        /\ waitSet = {}


Next == \E t \in RunningThreads: \/ /\ t \in Producers
     	     	 		    /\ Put(t,t)
				 \/ /\ t \in Consumers
				    /\ Get(t)

-----------------------------------------------------------------------------
TypeInv == /\ buffer \in Seq(Producers)
	   /\ Len(buffer) \in 0..BufCapacity
	   /\ waitSet \subseteq (Producers \cup Consumers)

Invariant == waitSet # (Producers \cup Consumers)


=============================================================================