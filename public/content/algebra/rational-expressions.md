# Rational Expressions

## Introduction

A **rational expression** is a fraction where the numerator and/or denominator are polynomials:

$$\frac{P(x)}{Q(x)} \quad \text{where } Q(x) \neq 0$$

Just like with fractions, we can simplify, add, subtract, multiply, and divide rational expressions.

**Key rule:** The denominator can **never** be zero. Any value that makes $Q(x) = 0$ is **excluded from the domain**.

## Domain Restrictions

For $\frac{x+3}{x-5}$, the denominator $x - 5 = 0$ when $x = 5$.
**Domain:** All real numbers except $x = 5$

For $\frac{x}{x^2 - 4}$, factor the denominator: $(x+2)(x-2) = 0$ when $x = -2$ or $x = 2$.
**Domain:** All real numbers except $x = -2, 2$

## Simplifying Rational Expressions

Factor numerator and denominator, then cancel common factors.

**Example 1:** $\frac{x^2 - 9}{x^2 + 5x + 6}$

Factor:
- Numerator: $(x+3)(x-3)$
- Denominator: $(x+2)(x+3)$

$$\frac{(x+3)(x-3)}{(x+2)(x+3)} = \frac{x-3}{x+2}, \quad x \neq -3$$

**Important:** The restriction $x \neq -3$ still applies to the simplified form, even though it cancels out!

**Example 2:** $\frac{2x^2 - 8}{4x + 8}$

Factor:
- Numerator: $2(x^2 - 4) = 2(x+2)(x-2)$
- Denominator: $4(x+2)$

$$\frac{2(x+2)(x-2)}{4(x+2)} = \frac{2(x-2)}{4} = \frac{x-2}{2}, \quad x \neq -2$$

## Multiplying Rational Expressions

**Steps:**
1. Factor everything
2. Cancel any factor that appears in both numerators and denominators
3. Multiply the remaining numerators and denominators

**Example:** $\frac{x^2 - 4}{x + 2} \times \frac{x^2 + 5x + 6}{x^2 - 9}$

**Step 1:** Factor:
$$= \frac{(x+2)(x-2)}{x+2} \times \frac{(x+2)(x+3)}{(x+3)(x-3)}$$

**Step 2:** Cancel:
$$= \frac{(x+2)(x-2)}{1} \times \frac{(x+2)}{1}$$

Wait, let me redo this more carefully:

$$= \frac{(x+2)(x-2)}{x+2} \times \frac{(x+2)(x+3)}{(x+3)(x-3)}$$

Cancel $x+2$ from first numerator and first denominator:
$$= \frac{(x-2)}{1} \times \frac{(x+2)(x+3)}{(x+3)(x-3)}$$

Cancel $x+3$ from second numerator and second denominator:
$$= (x-2) \times \frac{x+2}{x-3}$$

$$= \frac{(x-2)(x+2)}{x-3}, \quad x \neq -2, -3, 2$$

## Dividing Rational Expressions

**To divide by a fraction, multiply by its reciprocal.**

**Example:** $\frac{x+1}{x-3} \div \frac{x^2 - 1}{x+2}$

**Step 1:** Take reciprocal of second fraction: $\frac{x+2}{x^2 - 1}$

**Step 2:** Factor $x^2 - 1 = (x+1)(x-1)$

$$= \frac{x+1}{x-3} \times \frac{x+2}{(x+1)(x-1)}$$

**Step 3:** Cancel $(x+1)$:
$$= \frac{1}{x-3} \times \frac{x+2}{x-1} = \frac{x+2}{(x-3)(x-1)}$$

**Domain restrictions:** $x \neq 3, -1, 1, -2$

## Adding and Subtracting Rational Expressions

**Same denominator:** Just add/subtract numerators, keep denominator.

$$\frac{3x}{x+2} + \frac{5x}{x+2} = \frac{8x}{x+2}$$

**Different denominators:** Find the LCD (Least Common Denominator), then combine.

**Example:** $\frac{3}{x+2} + \frac{5}{x-1}$

**Step 1:** LCD = $(x+2)(x-1)$

**Step 2:** Rewrite each fraction with the LCD:
$$\frac{3(x-1)}{(x+2)(x-1)} + \frac{5(x+2)}{(x+2)(x-1)}$$

**Step 3:** Combine:
$$= \frac{3x - 3 + 5x + 10}{(x+2)(x-1)} = \frac{8x + 7}{(x+2)(x-1)}$$

## Complex Rational Expressions

A **complex rational expression** has fractions in its numerator or denominator.

**Example:**
$$\frac{\frac{1}{x} + \frac{1}{y}}{\frac{1}{x} - \frac{1}{y}}$$

**Method:** Combine terms in numerator and denominator separately, then divide.

**Step 1:** Combine numerator: $\frac{1}{x} + \frac{1}{y} = \frac{y+x}{xy}$

**Step 2:** Combine denominator: $\frac{1}{x} - \frac{1}{y} = \frac{y-x}{xy}$

**Step 3:** Divide:
$$\frac{\frac{y+x}{xy}}{\frac{y-x}{xy}} = \frac{y+x}{xy} \times \frac{xy}{y-x} = \frac{y+x}{y-x}$$

## Solving Rational Equations

**Key:** Multiply both sides by the LCD to clear denominators. **Always check for extraneous solutions!**

**Example:** $\frac{2}{x} + \frac{3}{x+1} = \frac{5}{x(x+1)}$

**Step 1:** LCD = $x(x+1)$

**Step 2:** Multiply by LCD:
$$2(x+1) + 3x = 5$$

**Step 3:** Solve:
$$2x + 2 + 3x = 5$$
$$5x = 3$$
$$x = \frac{3}{5}$$

**Step 4:** Check — does $x = 3/5$ make any denominator zero?
- $x = 3/5 \neq 0$ ✓
- $x + 1 = 8/5 \neq 0$ ✓
- $x(x+1) \neq 0$ ✓

**Solution:** $x = 3/5$

## Practice Problems

1. **Simplify:** $\frac{x^2 - x - 6}{x^2 - 9}$
   <details><summary>Answer</summary>$\frac{(x-3)(x+2)}{(x-3)(x+3)} = \frac{x+2}{x+3}, \quad x \neq 3, -3$</details>

2. **Multiply:** $\frac{x+4}{x-2} \times \frac{x^2 - 4}{x+4}$
   <details><summary>Answer</summary>$\frac{x+4}{x-2} \times \frac{(x+2)(x-2)}{x+4} = x+2, \quad x \neq 2, -4$</details>

3. **Divide:** $\frac{x-3}{x+5} \div \frac{x^2 - 9}{x+5}$
   <details><summary>Answer</summary>$\frac{x-3}{x+5} \times \frac{x+5}{(x+3)(x-3)} = \frac{1}{x+3}, \quad x \neq 3, -5, -3$</details>

4. **Add:** $\frac{2}{x+3} + \frac{1}{x-2}$
   <details><summary>Answer</summary>$\frac{2(x-2) + 1(x+3)}{(x+3)(x-2)} = \frac{3x-1}{(x+3)(x-2)}$</details>

5. **Solve:** $\frac{2}{x} + \frac{1}{2} = \frac{3}{2x}$
   <details><summary>Answer</summary>
   LCD = $2x$: $4 + x = 3$, so $x = -1$
   Check: $x = -1$ makes no denominator zero ✓
   </details>

## Key Takeaways

- **Domain restrictions** are critical — denominator can never be zero
- **Simplify** by factoring and canceling common factors
- **Multiply:** factor, cancel, multiply
- **Divide:** multiply by the reciprocal
- **Add/Subtract:** find LCD, combine numerators
- **Solve rational equations:** clear denominators, check for extraneous solutions
