# Quadratic Equations

## Introduction

A **quadratic equation** is any equation that can be written in the form:

$$ax^2 + bx + c = 0$$

where $a$, $b$, and $c$ are constants, and $a \neq 0$.

The **solutions** (or **roots**) of a quadratic equation are the values of $x$ that make the equation true.

## Methods for Solving Quadratics

There are four main methods, from simplest to most powerful:

1. Factoring (if possible)
2. Square Root Method (if no linear term)
3. Completing the Square (always works)
4. Quadratic Formula (always works)

## Method 1: Factoring

**When to use:** When the quadratic factors nicely.

**Example:** $x^2 + 5x + 6 = 0$

Factor: $(x + 2)(x + 3) = 0$

Set each factor to zero:
- $x + 2 = 0 \Rightarrow x = -2$
- $x + 3 = 0 \Rightarrow x = -3$

**Solutions:** $x = -2, -3$

**Checking:** $(-2)^2 + 5(-2) + 6 = 4 - 10 + 6 = 0$ ✓

## Method 2: Square Root Method

**When to use:** When the equation is in the form $ax^2 + c = 0$ (no linear term).

**Example:** $x^2 - 9 = 0$

$$x^2 = 9$$
$$x = \pm\sqrt{9}$$
$$x = \pm 3$$

**Example:** $(x - 2)^2 = 7$

$$x - 2 = \pm\sqrt{7}$$
$$x = 2 \pm \sqrt{7}$$

## Method 3: Completing the Square

**When to use:** Always works, especially when factoring is difficult.

**Steps:**
1. Make sure $a = 1$ (divide by $a$ if needed)
2. Move $c$ to the right side
3. Take half of $b$, square it, add to both sides
4. Factor the left as a perfect square
5. Take square root of both sides

**Example:** $x^2 + 6x - 4 = 0$

**Step 1:** $a = 1$ ✓

**Step 2:** $x^2 + 6x = 4$

**Step 3:** Half of $b = 6 \Rightarrow 3 \Rightarrow 3^2 = 9$

$$x^2 + 6x + 9 = 4 + 9$$
$$(x + 3)^2 = 13$$

**Step 4:** $x + 3 = \pm\sqrt{13}$

**Step 5:** $x = -3 \pm \sqrt{13}$

**Solutions:** $x = -3 + \sqrt{13}$, $x = -3 - \sqrt{13}$

## Method 4: Quadratic Formula

**The nuclear option — always works!**

For $ax^2 + bx + c = 0$:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

The expression under the square root, $b^2 - 4ac$, is called the **discriminant**.

**Example:** $2x^2 + 5x - 3 = 0$

$a = 2$, $b = 5$, $c = -3$

$$x = \frac{-5 \pm \sqrt{25 - 4(2)(-3)}}{2(2)}$$
$$= \frac{-5 \pm \sqrt{25 + 24}}{4}$$
$$= \frac{-5 \pm \sqrt{49}}{4}$$
$$= \frac{-5 \pm 7}{4}$$

$x = \frac{-5 + 7}{4} = \frac{2}{4} = \frac{1}{2}$ or $x = \frac{-5 - 7}{4} = \frac{-12}{4} = -3$

**Solutions:** $x = \frac{1}{2}, -3$

## The Discriminant: $b^2 - 4ac$

The discriminant tells you the **nature** of the solutions before you solve:

| Discriminant | Value | Meaning | Solutions |
|---------------|-------|---------|-----------|
| $b^2 - 4ac > 0$ | Positive | Two distinct real roots | Parabola crosses x-axis twice |
| $b^2 - 4ac = 0$ | Zero | One repeated real root | Parabola touches x-axis |
| $b^2 - 4ac < 0$ | Negative | Two complex roots | Parabola stays above or below x-axis |

**Example:** $x^2 + 4x + 5 = 0$
- $b^2 - 4ac = 16 - 20 = -4 < 0$
- Two complex roots: $x = \frac{-4 \pm 2i}{2} = -2 \pm i$

## Graphing Quadratics: The Parabola

The graph of $y = ax^2 + bx + c$ is a **parabola**.

- **Opens upward** ($a > 0$): minimum point at vertex
- **Opens downward** ($a < 0$): maximum point at vertex

**Vertex form:** $y = a(x - h)^2 + k$ where $(h, k)$ is the vertex.

**Finding the vertex:** $x = -\frac{b}{2a}$

**Example:** $y = 2x^2 + 8x + 3$
- $x = -\frac{8}{2(2)} = -2$
- $y = 2(-2)^2 + 8(-2) + 3 = 8 - 16 + 3 = -5$
- Vertex: $(-2, -5)$, opens upward (minimum)

## Word Problem Example

**Problem:** A ball is thrown upward with initial velocity 20 m/s from a height of 5 m. Its height is given by $h(t) = -5t^2 + 20t + 5$. When does the ball hit the ground?

**Solution:** Set $h = 0$:
$$-5t^2 + 20t + 5 = 0$$
Divide by $-5$: $t^2 - 4t - 1 = 0$

$$t = \frac{4 \pm \sqrt{16 + 4}}{2} = \frac{4 \pm \sqrt{20}}{2} = \frac{4 \pm 2\sqrt{5}}{2} = 2 \pm \sqrt{5}$$

$t \approx 2 + 2.236 = 4.236$ seconds (positive solution)

The ball hits the ground after approximately **4.24 seconds**.

## Practice Problems

1. **Solve by factoring:** $x^2 - 3x - 10 = 0$
   <details><summary>Answer</summary>$(x - 5)(x + 2) = 0$, so $x = 5$ or $x = -2$</details>

2. **Solve using square root:** $3x^2 = 12$
   <details><summary>Answer</summary>$x^2 = 4$, so $x = \pm 2$</details>

3. **Solve by completing the square:** $x^2 + 4x + 1 = 0$
   <details><summary>Answer</summary>$(x+2)^2 = 3$, so $x = -2 \pm \sqrt{3}$</details>

4. **Solve using quadratic formula:** $x^2 + 2x + 5 = 0$
   <details><summary>Answer</summary>$x = \frac{-2 \pm \sqrt{4-20}}{2} = \frac{-2 \pm \sqrt{-16}}{2} = -1 \pm 2i$</details>

5. **Find the discriminant:** $4x^2 + 12x + 9 = 0$
   <details><summary>Answer</summary>$b^2 - 4ac = 144 - 144 = 0$ (one repeated real root)</details>

## Key Takeaways

- **Factoring** is fastest when it works cleanly
- **Square root** method for $ax^2 + c = 0$ (no linear term)
- **Completing the square** always works and reveals the vertex
- **Quadratic formula** is the universal method — memorize it!
- **Discriminant** tells you how many real solutions before you solve
- **Vertex** is at $x = -b/(2a)$ — the minimum/maximum point
