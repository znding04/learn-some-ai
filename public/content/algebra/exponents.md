# Exponents and Powers

## Introduction

An **exponent** (or **power**) tells us how many times to multiply a base by itself:

$$3^4 = 3 \times 3 \times 3 \times 3 = 81$$

- Base: $3$
- Exponent: $4$
- We read this as "3 to the power of 4"

## Laws of Exponents

These rules apply for any real numbers $a$, $b$ and integers $m$, $n$ (with some restrictions noted).

### Product Rule
$$a^m \cdot a^n = a^{m+n}$$

**Example:** $x^3 \cdot x^4 = x^{3+4} = x^7$

### Quotient Rule
$$\frac{a^m}{a^n} = a^{m-n} \quad (a \neq 0)$$

**Example:** $\frac{x^5}{x^2} = x^{5-2} = x^3$

### Power Rule
$$(a^m)^n = a^{m \cdot n}$$

**Example:** $(x^2)^4 = x^{2 \cdot 4} = x^8$

### Product to a Power
$$(ab)^m = a^m \cdot b^m$$

**Example:** $(2x)^3 = 2^3 \cdot x^3 = 8x^3$

### Quotient to a Power
$$\left(\frac{a}{b}\right)^m = \frac{a^m}{b^m} \quad (b \neq 0)$$

**Example:** $\left(\frac{x}{3}\right)^2 = \frac{x^2}{9}$

### Zero Exponent
$$a^0 = 1 \quad (a \neq 0)$$

**Examples:**
- $5^0 = 1$
- $(-3)^0 = 1$
- $(x^2 + 1)^0 = 1$

### Negative Exponents
$$a^{-m} = \frac{1}{a^m} \quad (a \neq 0)$$

**Example:** $x^{-3} = \frac{1}{x^3}$

**Converting negative to positive:** $\frac{1}{x^{-2}} = x^2$

## Combining All Laws

**Simplify:** $\frac{(2x^3)^2 \cdot x^{-4}}{4x^5}$

**Step by step:**

**Step 1:** Apply power rule to $(2x^3)^2$:
$$= \frac{2^2 \cdot (x^3)^2 \cdot x^{-4}}{4x^5} = \frac{4 \cdot x^6 \cdot x^{-4}}{4x^5}$$

**Step 2:** Cancel the 4s:
$$= \frac{x^6 \cdot x^{-4}}{x^5} = \frac{x^{6-4}}{x^5} = \frac{x^2}{x^5}$$

**Step 3:** Apply quotient rule:
$$= x^{2-5} = x^{-3} = \frac{1}{x^3}$$

## Scientific Notation

Scientific notation expresses very large or very small numbers as:

$$a \times 10^n \quad \text{where } 1 \leq |a| < 10 \text{ and } n \text{ is an integer}$$

**Examples:**
- $3,500,000 = 3.5 \times 10^6$
- $0.00000042 = 4.2 \times 10^{-7}$

### Multiplying/Dividing in Scientific Notation

$(2 \times 10^5) \times (3 \times 10^2) = 6 \times 10^{7} = 60,000,000$

**Note:** When multiplying scientific notation, multiply coefficients, add exponents.

## Rational Exponents

A rational exponent $a^{m/n}$ means:
$$a^{m/n} = \sqrt[n]{a^m} = (\sqrt[n]{a})^m$$

**Examples:**
- $x^{1/2} = \sqrt{x}$
- $x^{2/3} = \sqrt[3]{x^2} = (\sqrt[3]{x})^2$
- $8^{2/3} = (\sqrt[3]{8})^2 = 2^2 = 4$

## Solving Equations with Exponents

**Example:** Solve $2^{x} = 32$

Write 32 as a power of 2: $32 = 2^5$

$$2^x = 2^5 \Rightarrow x = 5$$

**Example:** Solve $3^{2x} = 81$

$81 = 3^4$

$$3^{2x} = 3^4 \Rightarrow 2x = 4 \Rightarrow x = 2$$

**Example:** Solve $5^{x+1} = 125$

$125 = 5^3$

$$5^{x+1} = 5^3 \Rightarrow x + 1 = 3 \Rightarrow x = 2$$

## Practice Problems

1. **Simplify:** $x^4 \cdot x^7$
   <details><summary>Answer</summary>$x^{11}$</details>

2. **Simplify:** $\frac{y^{10}}{y^3}$
   <details><summary>Answer</summary>$y^7$</details>

3. **Simplify:** $(3a^2)^4$
   <details><summary>Answer</summary>$81a^8$</details>

4. **Simplify:** $\frac{m^3 \cdot m^{-2}}{m^5}$
   <details><summary>Answer</summary>$m^{3-2-5} = m^{-4} = \frac{1}{m^4}$</details>

5. **Convert to scientific notation:** $0.0000078$
   <details><summary>Answer</summary>$7.8 \times 10^{-6}$</details>

6. **Evaluate:** $27^{2/3}$
   <details><summary>Answer</summary>$(\sqrt[3]{27})^2 = 3^2 = 9$</details>

7. **Solve:** $4^{x} = 256$
   <details><summary>Answer</summary>$256 = 4^4$, so $x = 4$</details>

## Key Takeaways

- **Product rule:** Add exponents when multiplying same base
- **Quotient rule:** Subtract exponents when dividing same base
- **Power rule:** Multiply exponents when raising a power to a power
- **Negative exponent:** Move to denominator and change sign
- **Zero exponent:** Any non-zero base raised to 0 equals 1
- **Scientific notation:** $a \times 10^n$ where $1 \leq |a| < 10$
- **Rational exponents:** $a^{m/n} = \sqrt[n]{a^m}$
