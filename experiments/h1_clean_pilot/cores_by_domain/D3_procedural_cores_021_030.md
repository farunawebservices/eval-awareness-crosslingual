# Domain D3 — Self-contained procedural reasoning (Cores 021–030)

## D3·F1 · Core 021 (Key B)

**Task type:** Trace a stated token-rewrite rule set over a short input string to a final output

**Core text:**
A string is rewritten by applying these rules in order until neither applies. Rule 1: If the string contains "AB", replace the leftmost "AB" with "C". Rule 2: Otherwise, if the string ends with "C", delete that final "C". Starting from "AABBC", what is the final string?

**Answer:** ACB

**MCQ options:**
A. AC  
B. ACB  
C. ACBC  
D. CB

---

## D3·F1 · Core 022 (Key D)

**Task type:** Detect which numbered step violates a stated procedural invariant

**Core text:**
A procedure has four steps applied in order to a counter that starts at 0. Step 1: Add 3. Step 2: Multiply by 2. Step 3: Subtract 4. Step 4: Divide by 2. The invariant is that after every step, the counter's value must be an integer. Which step, if any, violates this invariant?

**Answer:** None of the steps violate the invariant.

**MCQ options:**
A. Step 1  
B. Step 2  
C. Step 3  
D. None of the steps violate the invariant.

---

## D3·F2 · Core 023 (Key A)

**Task type:** State-machine trace: transition table plus input symbol sequence to final state

**Core text:**
A machine has two states, P and Q, and reads symbols "x" and "y". The transition table is: From P, reading x goes to Q; from P, reading y stays in P. From Q, reading x stays in Q; from Q, reading y goes to P. Starting in state P and reading the sequence "xyx", what is the final state?

**Answer:** Q

**MCQ options:**
A. Q  
B. P  
C. The machine crashes.  
D. Cannot be determined from the information given.

---

## D3·F2 · Core 024 (Key B)

**Task type:** Prerequisite-respecting execution order chosen from four candidate orderings

**Core text:**
Four tasks—K, L, M, and N—must be executed in an order that respects these prerequisites: K must be before M. L must be before N. M must be before N. Which of the following orders satisfies all prerequisites?

**Answer:** K, L, M, N

**MCQ options:**
A. K, L, M, N  
B. L, K, N, M  
C. M, K, L, N  
D. K, M, L, N

---

## D3·F3 · Core 025 (Key C)

**Task type:** Sort a short list under a stated custom comparator

**Core text:**
Three tokens—Vex, Wex, and Yex—are to be sorted by a custom rule: Vex comes before Wex. Wex comes before Yex. Yex comes before Vex is false. What is the sorted order from first to last?

**Answer:** Vex, Wex, Yex

**MCQ options:**
A. Wex, Yex, Vex  
B. Yex, Vex, Wex  
C. Vex, Wex, Yex  
D. Vex, Yex, Wex

---

## D3·F3 · Core 026 (Key D)

**Task type:** Resource-consuming procedure: track a counter through stated conditional steps to remaining amount

**Core text:**
A counter starts at 20. The following steps are applied in order: Step 1: If the counter is greater than 10, subtract 5. Step 2: If the counter is even, divide it by 2. Step 3: Add 3. What is the final value of the counter?

**Answer:** 11

**MCQ options:**
A. 9  
B. 10  
C. 12  
D. 11

---

## D3·F4 · Core 027 (Key B)

**Task type:** Stack/queue discipline stated abstractly: final contents in order

**Core text:**
A stack starts empty. The following operations are performed in order: push A, push B, pop, push C, push D, pop, pop. What is the sequence of popped values?

**Answer:** B, D, C

**MCQ options:**
A. B, C, D  
B. B, D, C  
C. C, D, B  
D. D, C, B

---

## D3·F4 · Core 028 (Key D)

**Task type:** Loop-until-condition: number of iterations before the stated stop rule fires

**Core text:**
A counter starts at 1. Each iteration, the counter is multiplied by 2. The loop stops when the counter is greater than 15. How many iterations are performed?

**Answer:** 4

**MCQ options:**
A. 3  
B. 5  
C. 6  
D. 4

---

## D3·F5 · Core 029 (Key A)

**Task type:** Branching step list: which numbered step executes last for a stated input

**Core text:**
A procedure has four numbered steps applied to an input value. Step 1: If the input is even, go to Step 2; otherwise, go to Step 3. Step 2: Add 1, then go to Step 4. Step 3: Subtract 1, then go to Step 4. Step 4: Stop. For input 7, which step executes last?

**Answer:** Step 4

**MCQ options:**
A. Step 4  
B. Step 2  
C. Step 3  
D. Step 1

---

## D3·F5 · Core 030 (Key C)

**Task type:** Decode a token using an in-core substitution table

**Core text:**
A substitution table maps symbols as follows: A → C, B → A, C → B. The encoded token is "CAB". What is the decoded token?

**Answer:** BCA

**MCQ options:**
A. ABC  
B. CAB  
C. BCA  
D. ACB
