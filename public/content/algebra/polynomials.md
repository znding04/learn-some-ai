# Polynomials

## Introduction

A **polynomial** is an expression consisting of variables and coefficients, combined using addition, subtraction, and multiplication (but never division by a variable).

**Examples:**
- $3x^2 + 2x - 5$ (quadratic polynomial)
- $x^3 - 4x^2 + x + 7$ (cubic polynomial)
- $5x - 3$ (linear polynomial)
- $8$ (constant polynomial)

**Not polynomials:**
- $\frac{1}{x}$ (division by variable)
- $x^{1/2}$ (fractional exponent)
- $\sqrt{x}$ (radical)

## Polynomial Vocabulary

For the polynomial $4x^3 + 3x^2 - 2x + 7$:

| Term | Coefficient | Variable Part | Degree |
|------|------------|--------------|--------|
| $4x^3$ | $4$ | $x^3$ | 3 |
| $3x^2$ | $3$ | $x^2$ | 2 |
| $-2x$ | $-2$ | $x$ | 1 |
| $7$ | $7$ | (none) | 0 |

- **Degree** of the polynomial: the highest exponent = $3$ (cubic)
- **Leading coefficient**: the coefficient of the highest-degree term = $4$
- **Constant term**: the term with no variable = $7$

## Classifying by Degree

| Degree | Name | Example |
|--------|------|---------|
| 0 | Constant | $5$ |
| 1 | Linear | $3x - 2$ |
| 2 | Quadratic | $x^2 + 4x - 1$ |
| 3 | Cubic | $2x^3 - x^2 + 3$ |
| 4 | Quartic | $x^4 + 2x^3 - x + 5$ |
| 5 | Quintic | $x^5 - 3x^3 + 2x$ |

## Adding and Subtracting Polynomials

**Add:** $(3x^2 + 2x - 5) + (x^2 - 3x + 4)$

Combine like terms (same variable and same exponent):
$$= 3x^2 + x^2 + 2x - 3x - 5 + 4$$
$$= 4x^2 - x - 1$$

**Subtract:** $(4x^3 + 2x - 7) - (x^3 - 3x^2 + 5)$

Distribute the negative, then combine:
$$= 4x^3 + 2x - 7 - x^3 + 3x^2 - 5$$
$$= 3x^3 + 3x^2 + 2x - 12$$

## Multiplying Polynomials

### Multiplying by a Monomial

$(3x^2)(2x^3 - 4x + 5)$

Multiply each term by $3x^2$:
$$= 6x^5 - 12x^3 + 15x^2$$

### FOIL Method (for two binomials)

$(a + b)(c + d) = ac + ad + bc + bd$

**Example:** $(x + 3)(x + 5)$

$$= x^2 + 5x + 3x + 15$$
$$= x^2 + 8x + 15$$

### Grid Method (for any two polynomials)

**Example:** $(2x + 1)(x^2 - 3x + 4)$

| $\times$ | $x^2$ | $-3x$ | $+4$ |
|----------|-------|-------|------|
| $2x$ | $2x^3$ | $-6x^2$ | $8x$ |
| $+1$ | $x^2$ | $-3x$ | $4$ |

Add all results:
$$2x^3 - 6x^2 + 8x + x^2 - 3x + 4$$
$$= 2x^3 - 5x^2 + 5x + 4$$

## Special Products

### Square of a binomial

$(a + b)^2 = a^2 + 2ab + b^2$

$(a - b)^2 = a^2 - 2ab + b^2$

**Example:** $(x + 3)^2 = x^2 + 6x + 9$

### Product of sum and difference

$(a + b)(a - b) = a^2 - b^2$ (difference of squares)

**Example:** $(x + 4)(x - 4) = x^2 - 16$

## Dividing Polynomials

### Dividing by a monomial

$$\frac{6x^3 + 9x^2 - 3x}{3x} = \frac{6x^3}{3x} + \frac{9x^2}{3x} - \frac{3x}{3x} = 2x^2 + 3x - 1$$

### Polynomial Long Division

Divide $x^2 + 4x + 4$ by $x + 2$:

```
        x + 2
x + 2 ) x^2 + 4x + 4
        x^2 + 2x
        --------
              2x + 4
              2x + 4
              ------
                   0
```

**Quotient:** $x + 2$, **Remainder:** $0$

So $\frac{x^2 + 4x + 4}{x + 2} = x + 2$

## Practice Problems

1. **Add:** $(2x^2 + 3x - 1) + (x^2 - x + 4)$
   <details><summary>Answer</summary>$3x^2 + 2x + 3$</details>

2. **Multiply using FOIL:** $(x - 7)(x + 3)$
   <details><summary>Answer</summary>$x^2 - 7x + 3x - 21 = x^2 - 4x - 21$</details>

3. **Multiply using grid:** $(x + 2)(x^2 - x + 3)$
   <details><summary>Answer</summary>$x^3 - x^2 + 3x + 2x^2 - 2x + 6 = x^3 + x^2 + x + 6$</details>

4. **Divide:** $\frac{12x^4 - 6x^3 + 3x^2}{3x^2}$
   <details><summary>Answer</summary>$4x^2 - 2x + 1$</details>

5. **Expand:** $(2x + 5)^2$
   <details><summary>Answer</summary>$4x^2 + 20x + 25$</details>

## Key Takeaways

- Polynomials combine variables and coefficients with $+$, $-$, $\times$ only
- Degree = highest exponent; classify as linear, quadratic, cubic, etc.
- Always combine **like terms** (same variable, same exponent)
- FOIL works for binomial × binomial; use grid method for larger polynomials
- $(a+b)^2 \neq a^2 + b^2$ — remember the middle term!
