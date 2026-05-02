# Factoring Polynomials

## Introduction

**Factoring** is the process of breaking down a polynomial into a product of simpler polynomials (its "factors"). If multiplication expands expressions, factoring does the reverse — it rewrites a polynomial as a product.

Factoring is essential for:
- Solving quadratic equations
- Simplifying rational expressions
- Finding roots and zeros of polynomials

## Factoring out the GCF (Greatest Common Factor)

**Always check for a GCF first!**

For $6x^3 + 9x^2 - 3x$:
- GCF of coefficients: $\gcd(6, 9, 3) = 3$
- GCF of variable parts: $x$ (minimum power of $x$ across all terms)
- GCF = $3x$

$$6x^3 + 9x^2 - 3x = 3x(2x^2 + 3x - 1)$$

**Verify by expanding:** $3x(2x^2) = 6x^3$, $3x(3x) = 9x^2$, $3x(-1) = -3x$ ✓

## Factoring Trinomials: $ax^2 + bx + c$

### Method 1: Guess and Check (AC Method)

For $x^2 + 5x + 6$:
1. Find two numbers that **multiply** to $c = 6$ and **add** to $b = 5$
2. Numbers: $2$ and $3$ (because $2 \times 3 = 6$ and $2 + 3 = 5$)
3. Write: $(x + 2)(x + 3)$

**For $2x^2 + 7x + 3$** (where $a \neq 1$):
1. Find two numbers that multiply to $a \times c = 2 \times 3 = 6$ and add to $7$
2. Numbers: $6$ and $1$ (because $6 \times 1 = 6$ and $6 + 1 = 7$)
3. Rewrite middle term: $2x^2 + 6x + x + 3$
4. Factor by grouping: $2x(x + 3) + 1(x + 3) = (x + 3)(2x + 1)$

### Method 2: AC Method (Systematic)

For $6x^2 + 5x - 4$:
1. $a = 6$, $b = 5$, $c = -4$
2. $ac = 6 \times (-4) = -24$
3. Find two numbers that multiply to $-24$ and add to $5$: $8$ and $-3$
4. Rewrite: $6x^2 + 8x - 3x - 4$
5. Group: $(6x^2 + 8x) + (-3x - 4)$
6. Factor: $2x(3x + 4) - 1(3x + 4) = (3x + 4)(2x - 1)$

## Difference of Squares

Pattern: $a^2 - b^2 = (a + b)(a - b)$

**Examples:**
- $x^2 - 9 = x^2 - 3^2 = (x + 3)(x - 3)$
- $4x^2 - 25 = (2x)^2 - 5^2 = (2x + 5)(2x - 5)$
- $x^4 - 16 = (x^2)^2 - 4^2 = (x^2 + 4)(x^2 - 4) = (x^2 + 4)(x + 2)(x - 2)$

**Not a difference of squares:**
- $x^2 + 9$ — this is a **sum of squares** and does NOT factor over the reals
- $x^2 + 4 = (x + 2i)(x - 2i)$ (requires complex numbers)

## Perfect Square Trinomials

Pattern: $a^2 + 2ab + b^2 = (a + b)^2$
Pattern: $a^2 - 2ab + b^2 = (a - b)^2$

**Examples:**
- $x^2 + 6x + 9 = (x + 3)^2$ (because $2 \times x \times 3 = 6x$)
- $x^2 - 10x + 25 = (x - 5)^2$ (because $2 \times x \times 5 = 10x$)
- $4x^2 + 4x + 1 = (2x + 1)^2$

**How to check:** Is the middle term equal to $2ab$?

## Sum/Difference of Cubes

### Difference of Cubes
$a^3 - b^3 = (a - b)(a^2 + ab + b^2)$

**Example:** $x^3 - 8 = x^3 - 2^3 = (x - 2)(x^2 + 2x + 4)$

### Sum of Cubes
$a^3 + b^3 = (a + b)(a^2 - ab + b^2)$

**Example:** $x^3 + 27 = x^3 + 3^3 = (x + 3)(x^2 - 3x + 9)$

## Factoring by Grouping

For polynomials with 4+ terms, try grouping:

$ax + ay + bx + by = a(x + y) + b(x + y) = (x + y)(a + b)$

**Example:** $2x^2 + 4x + 3x + 6$
1. Group: $(2x^2 + 4x) + (3x + 6)$
2. Factor each: $2x(x + 2) + 3(x + 2)$
3. Factor out $(x + 2)$: $(x + 2)(2x + 3)$

## Factoring Strategy (Which Method to Use?)

1. **Is there a GCF?** → Factor it out first
2. **Two terms only?**
   - $a^2 - b^2$ → Difference of squares
   - $a^3 - b^3$ → Difference of cubes
   - $a^3 + b^3$ → Sum of cubes
3. **Three terms?**
   - Looks like $(a \pm b)^2$? → Perfect square trinomial
   - Otherwise → AC method / guess and check
4. **Four terms?** → Try grouping

## Practice Problems

1. **Factor out the GCF:** $12x^4 - 8x^3 + 4x^2$
   <details><summary>Answer</summary>$4x^2(3x^2 - 2x + 1)$</details>

2. **Factor:** $x^2 + 7x + 12$
   <details><summary>Answer</summary>$(x + 3)(x + 4)$</details>

3. **Factor:** $2x^2 + 5x - 3$
   <details><summary>Answer</summary>$(2x - 1)(x + 3)$</details>

4. **Factor:** $x^2 - 64$
   <details><summary>Answer</summary>$(x + 8)(x - 8)$</details>

5. **Factor:** $x^2 + 6x + 9$
   <details><summary>Answer</summary>$(x + 3)^2$</details>

6. **Factor:** $8x^3 + 27$
   <details><summary>Answer</summary>$(2x + 3)(4x^2 - 6x + 9)$</details>

## Key Takeaways

- **Always factor out the GCF first** before trying other methods
- $a^2 - b^2$ factors; $a^2 + b^2$ does NOT (over reals)
- "Two terms, both cubes" → sum of cubes or difference of cubes
- "Three terms, perfect square?" → check if $b = 2ab$
- When in doubt, use the AC method: find $ac$, split $b$, group
