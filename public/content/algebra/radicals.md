# Radicals and Square Roots

## Introduction

A **radical** is a symbol that asks us to find the root of a number. The most common is the **square root**:

$$\sqrt{9} = 3 \quad \text{because } 3^2 = 9$$

We also have **cube roots**, **fourth roots**, etc.:
- $\sqrt[3]{8} = 2$ because $2^3 = 8$
- $\sqrt[4]{16} = 2$ because $2^4 = 16$
- $\sqrt[5]{32} = 2$ because $2^5 = 32$

## Square Roots: Perfect Squares

| $n$ | $n^2$ | $\sqrt{n^2}$ |
|-----|-------|-------------|
| 1 | 1 | 1 |
| 2 | 4 | 2 |
| 3 | 9 | 3 |
| 4 | 16 | 4 |
| 5 | 25 | 5 |
| 6 | 36 | 6 |
| 7 | 49 | 7 |
| 8 | 64 | 8 |
| 9 | 81 | 9 |
| 10 | 100 | 10 |
| 11 | 121 | 11 |
| 12 | 144 | 12 |

## Simplifying Radicals

### Rule: $\sqrt{ab} = \sqrt{a} \cdot \sqrt{b}$

**Example:** $\sqrt{50} = \sqrt{25 \cdot 2} = \sqrt{25} \cdot \sqrt{2} = 5\sqrt{2}$

**Key:** Look for the largest perfect square factor.

**Example:** $\sqrt{72}$
- $72 = 36 \times 2$
- $\sqrt{72} = \sqrt{36} \cdot \sqrt{2} = 6\sqrt{2}$

**Example:** $\sqrt{200}$
- $200 = 100 \times 2$
- $\sqrt{200} = \sqrt{100} \cdot \sqrt{2} = 10\sqrt{2}$

### Simplifying $\sqrt[n]{a}$ (nth roots)

$$\sqrt[3]{54} = \sqrt[3]{27 \cdot 2} = \sqrt[3]{27} \cdot \sqrt[3]{2} = 3\sqrt[3]{2}$$

## Adding and Subtracting Radicals

**Only "like radicals" can be added.** Like radicals have the same radicand (the number under the root).

$$3\sqrt{2} + 5\sqrt{2} = 8\sqrt{2}$$

**But:** $3\sqrt{2} + 5\sqrt{3}$ cannot be combined (different radicands)

**Simplify first, then combine:**

**Example:** $\sqrt{8} + \sqrt{18} + \sqrt{2}$
- $\sqrt{8} = 2\sqrt{2}$
- $\sqrt{18} = 3\sqrt{2}$
- $\sqrt{2} = \sqrt{2}$

Total: $2\sqrt{2} + 3\sqrt{2} + \sqrt{2} = 6\sqrt{2}$

## Multiplying Radicals

### Rule: $\sqrt{a} \cdot \sqrt{b} = \sqrt{ab}$

**Example:** $\sqrt{3} \cdot \sqrt{12} = \sqrt{36} = 6$

**Example with variables:** $\sqrt{x} \cdot \sqrt{x} = \sqrt{x^2} = x$ (for $x \geq 0$)

**Example:** $\sqrt{2x} \cdot \sqrt{8x}$
$$= \sqrt{16x^2} = 4|x|$$

### Special products with radicals

$(a + b)(a - b) = a^2 - b^2$

**Example:** $(3 + \sqrt{5})(3 - \sqrt{5}) = 9 - 5 = 4$

$(a + b)^2 = a^2 + 2ab + b^2$

**Example:** $(1 + \sqrt{3})^2 = 1 + 2\sqrt{3} + 3 = 4 + 2\sqrt{3}$

## Dividing Radicals (Rationalizing Denominators)

### The Goal: No radicals in the denominator

**Example:** $\frac{1}{\sqrt{2}}$

Multiply top and bottom by $\sqrt{2}$:
$$= \frac{\sqrt{2}}{\sqrt{2} \cdot \sqrt{2}} = \frac{\sqrt{2}}{2}$$

**Example:** $\frac{3}{1 + \sqrt{5}}$

Multiply by the conjugate $(1 - \sqrt{5})$:
$$= \frac{3(1 - \sqrt{5})}{(1 + \sqrt{5})(1 - \sqrt{5})} = \frac{3(1 - \sqrt{5})}{1 - 5} = \frac{3(1 - \sqrt{5})}{-4} = \frac{3(\sqrt{5} - 1)}{4}$$

## Rationalizing Binomial Denominators

For $\frac{5}{3 + \sqrt{2}}$:

Multiply by the conjugate $\frac{3 - \sqrt{2}}{3 - \sqrt{2}}$:
$$= \frac{5(3 - \sqrt{2})}{9 - 2} = \frac{5(3 - \sqrt{2})}{7} = \frac{15 - 5\sqrt{2}}{7}$$

## Radical Equations

**To solve:** Isolate the radical, then square both sides. **Check for extraneous solutions!**

**Example:** $\sqrt{x + 3} = 5$

**Step 1:** Already isolated: $\sqrt{x + 3} = 5$

**Step 2:** Square both sides:
$$x + 3 = 25$$
$$x = 22$$

**Step 3:** Check: $\sqrt{22 + 3} = \sqrt{25} = 5$ ✓

**Example with two radicals:** $\sqrt{x + 1} + \sqrt{x - 1} = 3$

**Step 1:** Isolate one radical: $\sqrt{x + 1} = 3 - \sqrt{x - 1}$

**Step 2:** Square both sides:
$$x + 1 = 9 - 6\sqrt{x - 1} + (x - 1)$$
$$x + 1 = x + 8 - 6\sqrt{x - 1}$$
$$1 = 8 - 6\sqrt{x - 1}$$
$$6\sqrt{x - 1} = 7$$
$$\sqrt{x - 1} = \frac{7}{6}$$

**Step 3:** Square again:
$$x - 1 = \frac{49}{36}$$
$$x = \frac{85}{36}$$

**Step 4:** Check: $\sqrt{85/36 + 1} + \sqrt{85/36 - 1} = \sqrt{121/36} + \sqrt{49/36} = 11/6 + 7/6 = 18/6 = 3$ ✓

## Practice Problems

1. **Simplify:** $\sqrt{75}$
   <details><summary>Answer</summary>$\sqrt{25 \cdot 3} = 5\sqrt{3}$</details>

2. **Simplify:** $\sqrt[3]{54}$
   <details><summary>Answer</summary>$\sqrt[3]{27 \cdot 2} = 3\sqrt[3]{2}$</details>

3. **Add:** $2\sqrt{5} + 3\sqrt{20} - \sqrt{45}$
   <details><summary>Answer</summary>$2\sqrt{5} + 3(2\sqrt{5}) - 3\sqrt{5} = 2\sqrt{5} + 6\sqrt{5} - 3\sqrt{5} = 5\sqrt{5}$</details>

4. **Multiply:** $(2 + \sqrt{3})(4 - \sqrt{3})$
   <details><summary>Answer</summary>$8 - 2\sqrt{3} + 4\sqrt{3} - 3 = 5 + 2\sqrt{3}$</details>

5. **Rationalize:** $\frac{6}{2 - \sqrt{3}}$
   <details><summary>Answer</summary>$\frac{6(2 + \sqrt{3})}{4 - 3} = 6(2 + \sqrt{3}) = 12 + 6\sqrt{3}$</details>

6. **Solve:** $\sqrt{2x - 5} = 3$
   <details><summary>Answer</summary>$2x - 5 = 9 \Rightarrow x = 7$. Check: $\sqrt{14-5} = \sqrt{9} = 3$ ✓</details>

## Key Takeaways

- $\sqrt{ab} = \sqrt{a} \cdot \sqrt{b}$ — break radicands into perfect square factors
- Only combine **like radicals** (same radicand)
- Multiply radicals: $\sqrt{a} \cdot \sqrt{b} = \sqrt{ab}$
- **Always rationalize** denominators (no radicals in the denominator)
- When solving radical equations: isolate, square, **check for extraneous solutions**
- $(a + b)(a - b) = a^2 - b^2$ is useful for rationalizing binomial denominators
