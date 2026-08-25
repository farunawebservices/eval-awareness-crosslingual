# Domain D5 — Scheduling and constraint planning (Cores 041–050)

## D5·F1 · Core 041 (Key B)

**Task type:** Batching under a stated capacity limit: minimum number of batches

**Core text:**
Six items must be shipped in batches. Each batch can hold at most 4 items. The items have sizes: 2, 3, 1, 4, 2, and 3 units. What is the minimum number of batches needed to ship all items?

**Answer:** 3 batches

**MCQ options:**
A. 2 batches  
B. 3 batches  
C. 4 batches  
D. 5 batches

---

## D5·F1 · Core 042 (Key D)

**Task type:** Which single constraint, if dropped, makes the stated plan feasible

**Core text:**
A schedule has three constraints: Constraint 1: Task K must be in slot 1. Constraint 2: Task L must be in slot 3. Constraint 3: Tasks K and L cannot be in adjacent slots. The current plan places K in slot 1 and L in slot 3, but this violates one constraint. Which single constraint, if dropped, makes the plan feasible?

**Answer:** Constraint 3

**MCQ options:**
A. Constraint 1  
B. Constraint 2  
C. All constraints are already satisfied.  
D. Constraint 3

---

## D5·F2 · Core 043 (Key B)

**Task type:** Slot assignment with pairwise exclusion constraints (who occupies which slot)

**Core text:**
Four tasks—K, L, M, and N—must be assigned to four slots numbered 1 to 4, one task per slot. Constraints: K cannot be in slot 1. L cannot be in slot 2. M cannot be in slot 3. N cannot be in slot 4. Which assignment satisfies all constraints?

**Answer:** L, K, N, M

**MCQ options:**
A. K, L, M, N  
B. L, K, N, M  
C. M, N, K, L  
D. N, M, L, K

---

## D5·F2 · Core 044 (Key D)

**Task type:** Shift rotation satisfying a stated no-consecutive rule

**Core text:**
Three workers—Vex, Wex, and Yex—must cover three consecutive shifts. Each worker covers exactly one shift. Constraint: No worker can cover two consecutive shifts (this is automatically satisfied if each covers exactly one). Which rotation is valid?

**Answer:** Vex, Wex, Yex

**MCQ options:**
A. Vex, Vex, Wex  
B. Wex, Wex, Yex  
C. Yex, Vex, Vex  
D. Vex, Wex, Yex

---

## D5·F3 · Core 045 (Key C)

**Task type:** Earliest completion slot from stated durations plus precedence pairs

**Core text:**
Three tasks must be completed in order. Task K takes 2 slots. Task L takes 3 slots. Task M takes 1 slot. Task L cannot start until K finishes. Task M cannot start until L finishes. If K starts at slot 1, in which slot does M finish?

**Answer:** Slot 6

**MCQ options:**
A. Slot 4  
B. Slot 5  
C. Slot 6  
D. Slot 7

---

## D5·F3 · Core 046 (Key D)

**Task type:** Priority-queue service order with a stated tie-break (which item is third)

**Core text:**
Four items—K, L, M, and N—are in a priority queue. Priorities: K has priority 3, L has priority 1, M has priority 3, N has priority 2. Higher priority numbers are served first. When priorities are equal, items are served in alphabetical order. Which item is served third?

**Answer:** N

**MCQ options:**
A. K  
B. L  
C. M  
D. N

---

## D5·F4 · Core 047 (Key B)

**Task type:** Availability-block intersection over relative slot labels (only common window)

**Core text:**
Worker Vex is available in slots 2, 3, and 4. Worker Wex is available in slots 3, 4, and 5. Worker Yex is available in slots 4, 5, and 6. In which slot are all three workers available?

**Answer:** Slot 4

**MCQ options:**
A. Slot 3  
B. Slot 4  
C. Slot 5  
D. No slot has all three available.

---

## D5·F4 · Core 048 (Key D)

**Task type:** Deadline feasibility: which single task cannot finish under stated ordering rules

**Core text:**
Three tasks must finish by slot 5. Task K takes 2 slots. Task L takes 2 slots. Task M takes 2 slots. Task K must be first. Task L must be second. Task M must be third. If K starts at slot 1, which task cannot finish by slot 5?

**Answer:** Task M

**MCQ options:**
A. Task K  
B. Task L  
C. All tasks can finish by slot 5.  
D. Task M

---

## D5·F5 · Core 049 (Key A)

**Task type:** Double-booking detection: which resource pair conflicts in the stated grid

**Core text:**
A resource grid shows two resources, R1 and R2, across four slots. R1 is booked in slots 1, 2, and 3. R2 is booked in slots 2, 3, and 4. Which slot has a double-booking conflict?

**Answer:** Slot 2

**MCQ options:**
A. Slot 2  
B. Slot 1  
C. Slot 4  
D. No slot has a double-booking.

---

## D5·F5 · Core 050 (Key C)

**Task type:** Minimum total setup cost from a stated type-to-type setup penalty table

**Core text:**
A setup table shows costs: Type K to Type L costs 3 credits. Type L to Type M costs 2 credits. Type K to Type M costs 5 credits. A sequence K → L → M is to be processed. What is the total setup cost?

**Answer:** 5 credits

**MCQ options:**
A. 3 credits  
B. 7 credits  
C. 5 credits  
D. 10 credits
