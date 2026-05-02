# Systems of Equations

## Introduction

A **system of equations** is a set of two or more equations with the same variables. The solution to a system is the set of values that satisfy **all** equations simultaneously.

For example:
$$\begin{cases} 2x + y = 10 \\ x - y = 2 \end{cases}$$

We need to find values of $x$ and $y$ that make both equations true at the same time.

## Method 1: Substitution

**Best for:** When one variable is already isolated or easy to isolate.

**Steps:**
1. Solve one equation for one variable in terms of the others
2. Substitute that expression into the other equation(s)
3. Solve for the remaining variable
4. Back-substitute to find the first variable

### Example

Solve:
$$\begin{cases} y = 2x + 3 \\ 3x + y = 8 \end{cases}$$

**Step 1:** Equation 1 already has $y$ solved: $y = 2x + 3$

**Step 2:** Substitute into Equation 2:
$$3x + (2x + 3) = 8$$

**Step 3:** Solve:
$$5x + 3 = 8$$
$$5x = 5$$
$$x = 1$$

**Step 4:** Back-substitute:
$$y = 2(1) + 3 = 5$$

**Solution:** $(x, y) = (1, 5)$

**Verification:** $3(1) + 5 = 8$ ✓

## Method 2: Elimination (Addition/Subtraction)

**Best for:** When coefficients of one variable are opposites or can become opposites.

**Steps:**
1. Multiply equations by constants so one variable has equal (or opposite) coefficients
2. Add or subtract equations to eliminate that variable
3. Solve for the remaining variable
4. Back-substitute

### Example

Solve:
$$\begin{cases} 2x + y = 10 \\ x - y = 2 \end{cases}$$

**Step 1:** Notice the coefficients of $y$ are $+1$ and $-1$ — they are already opposites!

**Step 2:** Add the equations:
$$(2x + y) + (x - y) = 10 + 2$$
$$3x = 12$$
$$x = 4$$

**Step 3:** Back-substitute into $x - y = 2$:
$$4 - y = 2$$
$$y = 2$$

**Solution:** $(x, y) = (4, 2)$

## Method 3: Graphing

**Best for:** Visual understanding and checking.

Graph both equations on the same coordinate plane. The **intersection point** is the solution.

- If lines intersect at one point → **one solution** (consistent and independent)
- If lines are parallel → **no solution** (inconsistent)
- If lines coincide → **infinitely many solutions** (consistent and dependent)

## Types of Systems

| Type | What it means | Example |
|------|---------------|---------|
| One solution | Lines intersect at exactly one point | $y = 2x + 1$ and $y = -x + 4$ |
| No solution | Lines are parallel (never meet) | $y = 2x + 1$ and $y = 2x - 3$ |
| Infinitely many | Lines are the same | $y = 2x + 1$ and $2y = 4x + 2$ |

## Word Problem Example

**Problem:** Tickets to a play cost \$8 for adults and \$5 for children. A group of 12 people bought tickets for a total of \$78. How many adults and children?

**Setup:**
Let $a$ = number of adults, $c$ = number of children

$$\begin{cases} a + c = 12 \\ 8a + 5c = 78 \end{cases}$$

**Solve by substitution** (from first equation: $c = 12 - a$):

$$8a + 5(12 - a) = 78$$
$$8a + 60 - 5a = 78$$
$$3a = 18$$
$$a = 6$$

$$c = 12 - 6 = 6$$

**Answer:** 6 adults and 6 children.

## Practice Problems

1. **Solve by substitution:**
   $$\begin{cases} y = 3x - 7 \\ 2x + y = 4 \end{cases}$$
   <details><summary>Answer</summary>
   Substitute: $2x + (3x - 7) = 4 \Rightarrow 5x = 11 \Rightarrow x = 11/5$
   $y = 3(11/5) - 7 = 33/5 - 35/5 = -2/5$
   Solution: $(11/5, -2/5)$
   </details>

2. **Solve by elimination:**
   $$\begin{cases} 3x + 2y = 16 \\ x - 2y = 4 \end{cases}$$
   <details><summary>Answer</summary>
   Add: $4x = 20 \Rightarrow x = 5$
   Back-substitute: $5 - 2y = 4 \Rightarrow y = 1/2$
   Solution: $(5, 1/2)$
   </details>

3. **Solve by graphing (estimate):**
   $$\begin{cases} y = x + 2 \\ y = -2x + 5 \end{cases}$$
   <details><summary>Answer</summary>
   Set equal: $x + 2 = -2x + 5 \Rightarrow 3x = 3 \Rightarrow x = 1$
   $y = 1 + 2 = 3$
   Solution: $(1, 3)$
   </details>

## Key Takeaways

- Substitution works well when a variable is already isolated
- Elimination works well when coefficients match or oppose
- Graphing gives visual intuition about the solution type
- Always verify your solution in both original equations
