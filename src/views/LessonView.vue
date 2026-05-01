<template>
  <div class="max-w-5xl mx-auto px-6 py-12">
    <!-- Navigation -->
    <nav class="flex items-center justify-between mb-10">
      <router-link
        :to="backLink"
        class="text-stone-400 hover:text-stone-200 transition-colors text-sm inline-flex items-center gap-1"
      >
        ← {{ backLabel }}
      </router-link>
      <button
        @click="markComplete"
        class="text-sm font-medium px-4 py-2 rounded-lg transition-colors"
        :class="isCompleted ? 'bg-emerald-700 text-emerald-200 hover:bg-emerald-600' : 'bg-stone-700 text-stone-300 hover:bg-stone-600'"
      >
        {{ isCompleted ? '✓ Completed' : 'Mark as Complete' }}
      </button>
    </nav>

    <!-- Breadcrumb -->
    <div class="flex items-center gap-2 text-sm text-stone-500 mb-8">
      <router-link to="/" class="hover:text-stone-300">Home</router-link>
      <span>›</span>
      <router-link to="/topics" class="hover:text-stone-300">Topics</router-link>
      <span>›</span>
      <router-link :to="`/topic/${lesson?.topic}`" class="hover:text-stone-300">{{ topicTitle }}</router-link>
      <span>›</span>
      <span class="text-stone-300">{{ lesson?.title }}</span>
    </div>

    <!-- Lesson Header -->
    <header class="mb-10">
      <h1 class="text-4xl font-bold text-stone-50 mb-4">{{ lesson?.title }}</h1>
      <div class="flex flex-wrap items-center gap-4 text-sm text-stone-400">
        <span v-if="lesson?.estimatedTime" class="flex items-center gap-1">
          ⏱ {{ lesson.estimatedTime }}
        </span>
        <span v-if="lesson?.difficulty" class="px-2 py-0.5 rounded text-xs font-medium"
          :class="lesson.difficulty === 'beginner' ? 'bg-emerald-900 text-emerald-300' : lesson.difficulty === 'intermediate' ? 'bg-amber-900 text-amber-300' : 'bg-red-900 text-red-300'"
        >
          {{ lesson.difficulty }}
        </span>
        <span v-if="lesson?.prerequisites?.length" class="flex items-center gap-1">
          📋 Prerequisites: {{ lesson.prerequisites.join(', ') }}
        </span>
      </div>
    </header>

    <!-- Embedded video -->
    <div v-if="lesson?.videoUrl" class="mb-10">
      <div class="aspect-video rounded-2xl overflow-hidden bg-stone-800 border border-stone-700">
        <iframe
          :src="lesson.videoUrl"
          class="w-full h-full"
          frameborder="0"
          allowfullscreen
        />
      </div>
    </div>

    <!-- Lesson content -->
    <article
      class="prose prose-invert prose-stone max-w-none mb-12"
      v-html="renderedContent"
    />

    <!-- Resource links -->
    <section v-if="lesson?.resources?.length" class="mb-12">
      <h2 class="text-2xl font-semibold text-stone-200 mb-6">Curated Resources</h2>
      <div class="grid gap-4 md:grid-cols-2">
        <a
          v-for="resource in lesson.resources"
          :key="resource.url"
          :href="resource.url"
          target="_blank"
          rel="noopener noreferrer"
          class="flex items-start gap-3 bg-stone-800 border border-stone-700 rounded-xl p-4 hover:border-stone-500 transition-colors group"
        >
          <span class="text-2xl flex-shrink-0">
            {{ resource.type === 'video' ? '🎬' : resource.type === 'article' ? '📄' : '🔗' }}
          </span>
          <div class="min-w-0">
            <div class="flex items-center gap-2 mb-1">
              <span class="text-xs uppercase tracking-wider text-stone-500">{{ resource.type }}</span>
              <span v-if="resource.duration" class="text-xs text-stone-500">{{ resource.duration }}</span>
            </div>
            <h3 class="text-stone-100 font-medium group-hover:text-white transition-colors">{{ resource.title }}</h3>
            <p class="text-stone-500 text-xs mt-1 truncate">{{ resource.url }}</p>
          </div>
        </a>
      </div>
    </section>

    <!-- Navigation: prev / next -->
    <div class="flex items-center justify-between border-t border-stone-800 pt-8 mt-12">
      <router-link
        v-if="prevLesson"
        :to="`/lesson/${prevLesson.id}`"
        class="flex items-center gap-2 bg-stone-800 border border-stone-700 rounded-xl px-5 py-3 hover:border-stone-500 transition-colors"
      >
        <span class="text-stone-400">←</span>
        <div>
          <p class="text-xs text-stone-500 mb-0.5">Previous</p>
          <p class="text-stone-100 font-medium">{{ prevLesson.title }}</p>
        </div>
      </router-link>
      <div v-else />
      <router-link
        v-if="nextLesson"
        :to="`/lesson/${nextLesson.id}`"
        class="flex items-center gap-2 bg-stone-800 border border-stone-700 rounded-xl px-5 py-3 hover:border-stone-500 transition-colors text-right"
      >
        <div>
          <p class="text-xs text-stone-500 mb-0.5">Next</p>
          <p class="text-stone-100 font-medium">{{ nextLesson.title }}</p>
        </div>
        <span class="text-stone-400">→</span>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import MarkdownIt from 'markdown-it'
import katex from 'katex'

const route = useRoute()
const lessonId = computed(() => route.params.id)

// Complete lesson registry — in production this would come from content JSON
const lessonRegistry = {
  'algebra-variables': {
    id: 'algebra-variables', title: 'Variables and Expressions', topic: 'math-algebra', estimatedTime: '25min',
    difficulty: 'beginner', prerequisites: [], summary: 'Understanding what variables are.',
    videoUrl: '',
    resources: [
      { type: 'video', url: 'https://www.youtube.com/embed/v-7pz06wneI', title: 'Khan Academy — Variables and Expressions', duration: '10min' },
      { type: 'article', url: 'https://www.mathsisfun.com/definitions/variable.html', title: 'Math is Fun — Variable Definition' },
    ],
    content: `# Variables and Expressions

## Introduction

A **variable** is a symbol — usually a letter — that represents an unknown or changing value. Variables are the foundation of algebra and all higher mathematics.

For example, in the expression $3x + 5$, the letter $x$ is a variable. We don't know what $x$ is yet, but we know how it relates to other numbers.

## Why Use Variables?

Variables allow us to:
- Write **general rules** instead of specific numbers
- Express **relationships** between quantities
- Solve **unknowns** in equations

Instead of saying "double a number and add 7", we write: $2x + 7$

## Writing Expressions

Translate these phrases into algebraic expressions:

| Phrase | Expression |
|--------|-----------|
| A number increased by 5 | $x + 5$ |
| Twice a number | $2x$ |
| The difference of a number and 3 | $x - 3$ |
| Half of a number | $\\frac{x}{2}$ |

## Evaluating Expressions

Once we know the value of a variable, we can evaluate the expression:

If $x = 4$, then $3x + 5 = 3(4) + 5 = 12 + 5 = 17$

## Practice Problems

1. **If $a = 3$ and $b = 7$, evaluate $2a + b$**
   <details><summary>Answer</summary>$2(3) + 7 = 6 + 7 = 13$</details>

2. **Write an expression for "5 less than twice a number"**
   <details><summary>Answer</summary>$2x - 5$</details>

3. **If $y = 10$, what is $100 - 3y$?**
   <details><summary>Answer</summary>$100 - 30 = 70$</details>

## Key Takeaways

- Variables represent unknown values
- Expressions combine variables, numbers, and operations
- Evaluate by substituting known values and following order of operations
- $\\mathbb{R}$ (the set of all real numbers) is typically our universe of possible values
`
  },
  'algebra-equations': {
    id: 'algebra-equations', title: 'Linear Equations', topic: 'math-algebra', estimatedTime: '30min',
    difficulty: 'beginner', prerequisites: ['algebra-variables'], summary: 'Solving equations of the form ax + b = c.',
    videoUrl: '',
    resources: [
      { type: 'video', url: 'https://www.youtube.com/embed/LDiR4_JSfB8', title: 'Khan Academy — Linear Equations', duration: '12min' },
      { type: 'article', url: 'https://www.purplemath.com/modules/solvlin.htm', title: 'Purplemath — Solving Linear Equations' },
    ],
    content: `# Linear Equations

## What is a Linear Equation?

A linear equation is an equation where the highest power of any variable is 1. Its graph is always a straight line.

The standard form: $ax + b = c$ where $a \\neq 0$

## Solving One-Step Equations

If $x + 5 = 12$, subtract 5 from both sides:
$$x = 12 - 5 = 7$$

If $\\frac{x}{3} = 4$, multiply both sides by 3:
$$x = 4 \\times 3 = 12$$

## Solving Two-Step Equations

If $2x + 3 = 11$:
1. Subtract 3: $2x = 8$
2. Divide by 2: $x = 4$

**Rule:** Always do the same operation to both sides.

## Special Cases

- **No solution:** $x + 1 = x + 2$ → $1 = 2$ (impossible)
- **Infinite solutions:** $2x + 2 = 2(x + 1)$ → always true

## Practice Problems

1. **Solve: $5x - 3 = 22$**
   <details><summary>Answer</summary>$5x = 25$, so $x = 5$</details>

2. **Solve: $\\frac{2x}{7} = 6$**
   <details><summary>Answer</summary>$2x = 42$, so $x = 21$</details>

3. **Solve: $4(x - 1) = 12$**
   <details><summary>Answer</summary>$4x - 4 = 12$, so $4x = 16$, so $x = 4$</details>
`
  },
  'physics-motion': {
    id: 'physics-motion', title: 'Motion and Kinematics', topic: 'physics-mechanics', estimatedTime: '40min',
    difficulty: 'beginner', prerequisites: [], summary: 'Position, velocity, acceleration, and the equations of motion.',
    videoUrl: '',
    resources: [
      { type: 'video', url: 'https://www.youtube.com/embed/u_8oD8-OxC8', title: 'Khan Academy — One-Dimensional Motion', duration: '15min' },
      { type: 'article', url: 'https://physics.info/motion/', title: 'Physics.info — Motion' },
    ],
    content: `# Motion and Kinematics

## Describing Motion

Kinematics is the branch of physics that describes how objects move — without discussing *why* they move (that's dynamics).

### Key Quantities

| Symbol | Quantity | SI Unit |
|--------|----------|---------|
| $x$ or $s$ | Position | m |
| $v$ | Velocity | m/s |
| $a$ | Acceleration | m/s² |
| $t$ | Time | s |

## The Equations of Motion

For constant acceleration:

$$v = v_0 + at$$

$$x = x_0 + v_0 t + \\frac{1}{2}at^2$$

$$v^2 = v_0^2 + 2a(x - x_0)$$

## Practice Problems

1. **A car accelerates from rest at 3 m/s². How far does it travel in 5 seconds?**
   <details><summary>Answer</summary>$x = \\frac{1}{2}(3)(5^2) = 37.5$ m</details>

2. **A ball is thrown upward at 20 m/s. What is its velocity after 2 seconds? (take g = 10 m/s²)**
   <details><summary>Answer</summary>$v = 20 - 10(2) = 0$ m/s (at the top of its trajectory)</details>
`
  },
  'ml-intro': {
    id: 'ml-intro', title: 'What is Machine Learning?', topic: 'ai-ml', estimatedTime: '25min',
    difficulty: 'beginner', prerequisites: [], summary: 'Definitions, types of ML, and key concepts.',
    videoUrl: '',
    resources: [
      { type: 'video', url: 'https://www.youtube.com/embed/aircAruvnKk', title: '3Blue1Brown — What is Machine Learning?', duration: '10min' },
      { type: 'article', url: 'https://developers.google.com/machine-learning/intro-to-ml', title: 'Google — Introduction to Machine Learning' },
    ],
    content: `# What is Machine Learning?

## Definition

**Machine learning (ML)** is a subset of artificial intelligence where computers learn patterns from data — rather than being explicitly programmed with rules.

Traditional programming:
$$\\text{data} + \\text{program} \\rightarrow \\text{output}$$

Machine learning:
$$\\text{data} + \\text{output} \\rightarrow \\text{program}$$

## Three Types of ML

### 1. Supervised Learning
Learn from labeled examples (input → correct output).
- **Classification:** Predict a category (spam/not spam)
- **Regression:** Predict a continuous number (house price)

### 2. Unsupervised Learning
Find patterns in unlabeled data.
- **Clustering:** Group similar items (customer segmentation)
- **Dimensionality Reduction:** Compress features (PCA)

### 3. Reinforcement Learning
Learn by trial and error, receiving rewards or penalties.
- Used in robotics, game AI, recommendation systems.

## Key Concepts

- **Model:** The mathematical representation being learned
- **Training:** The process of adjusting model parameters to minimize error
- **Loss function:** Measures how wrong predictions are
- **Overfitting:** Model memorizes training data, fails on new data
- **Generalization:** Ability to perform well on unseen data

## Practice Reflection

1. **Would you use supervised or unsupervised learning to detect anomalous network traffic?**
   <details><summary>Answer</summary>Supervised (labeled normal/anomaly) if labels exist; unsupervised if no labels (anomaly detection / clustering approach).</details>
`
  },
}

// Map lesson IDs to registry entries
const registry = {
  'algebra-variables': lessonRegistry['algebra-variables'],
  'algebra-equations': lessonRegistry['algebra-equations'],
  'physics-motion': lessonRegistry['physics-motion'],
  'ml-intro': lessonRegistry['ml-intro'],
}

// All lessons for prev/next navigation
const allLessons = [
  lessonRegistry['algebra-variables'],
  lessonRegistry['algebra-equations'],
  lessonRegistry['physics-motion'],
  lessonRegistry['ml-intro'],
]

const lesson = computed(() => registry[lessonId.value] || null)

const topicTitle = computed(() => {
  if (!lesson.value) return ''
  if (lesson.value.topic === 'math-algebra') return 'Math — Algebra'
  if (lesson.value.topic === 'physics-mechanics') return 'Physics — Mechanics'
  if (lesson.value.topic === 'ai-ml') return 'AI / ML'
  return lesson.value.topic
})

const backLink = computed(() => lesson.value ? `/topic/${lesson.value.topic}` : '/topics')
const backLabel = computed(() => lesson.value ? topicTitle.value : 'Topics')

const lessonIndex = computed(() => allLessons.findIndex(l => l.id === lessonId.value))
const prevLesson = computed(() => lessonIndex.value > 0 ? allLessons[lessonIndex.value - 1] : null)
const nextLesson = computed(() => lessonIndex.value < allLessons.length - 1 ? allLessons[lessonIndex.value + 1] : null)

const isCompleted = ref(false)

onMounted(() => {
  try {
    const completed = JSON.parse(localStorage.getItem('wtl-completed') || '[]')
    isCompleted.value = completed.includes(lessonId.value)
  } catch {
    isCompleted.value = false
  }
})

function markComplete() {
  try {
    const completed = JSON.parse(localStorage.getItem('wtl-completed') || '[]')
    if (!isCompleted.value) {
      completed.push(lessonId.value)
    } else {
      const idx = completed.indexOf(lessonId.value)
      if (idx !== -1) completed.splice(idx, 1)
    }
    localStorage.setItem('wtl-completed', JSON.stringify(completed))
    isCompleted.value = !isCompleted.value
  } catch (e) {
    console.error('Failed to update completion status', e)
  }
}

// Markdown rendering with KaTeX support
const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
})

function renderMath(content) {
  // Block math: $$...$$
  content = content.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
    try {
      return `<div class="katex-display">${katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false })}</div>`
    } catch {
      return `<div class="katex-error">$$${tex}$$</div>`
    }
  })
  // Inline math: $...$
  content = content.replace(/\$([^\$\n]+?)\$/g, (_, tex) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false })
    } catch {
      return `$${tex}$`
    }
  })
  return content
}

const renderedContent = computed(() => {
  if (!lesson.value?.content) return ''
  const withMath = renderMath(lesson.value.content)
  return md.render(withMath)
})
</script>

<style>
/* KaTeX display math centering */
.katex-display {
  margin: 1.5rem 0;
  overflow-x: auto;
}

/* markdown-it generated styles */
.prose h1 { font-size: 2rem; font-weight: 700; color: #fafaf9; margin: 2rem 0 1rem; }
.prose h2 { font-size: 1.5rem; font-weight: 600; color: #e7e5e4; margin: 1.75rem 0 0.75rem; border-bottom: 1px solid #292524; padding-bottom: 0.5rem; }
.prose h3 { font-size: 1.25rem; font-weight: 600; color: #e7e5e4; margin: 1.5rem 0 0.5rem; }
.prose p { color: #d6d3d1; line-height: 1.8; margin: 1rem 0; }
.prose table { width: 100%; border-collapse: collapse; margin: 1.5rem 0; }
.prose th { background: #292524; color: #fafaf9; padding: 0.5rem 1rem; text-align: left; }
.prose td { border: 1px solid #292524; color: #d6d3d1; padding: 0.5rem 1rem; }
.prose tr:nth-child(even) td { background: #1c1917; }
.prose a { color: #6ee7b7; text-decoration: underline; }
.prose a:hover { color: #a7f3d0; }
.prose strong { color: #fafaf9; font-weight: 600; }
.prose em { color: #d6d3d1; font-style: italic; }
.prose ul, .prose ol { color: #d6d3d1; padding-left: 1.5rem; margin: 1rem 0; }
.prose li { margin: 0.25rem 0; line-height: 1.7; }
.prose code { background: #292524; color: #fca5a5; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem; }
.prose pre { background: #1c1917; border: 1px solid #292524; border-radius: 0.5rem; padding: 1rem; overflow-x: auto; margin: 1.5rem 0; }
.prose pre code { background: none; padding: 0; color: #d6d3d1; }
.prose blockquote { border-left: 4px solid #44403c; padding-left: 1rem; color: #78716c; margin: 1.5rem 0; font-style: italic; }
.prose hr { border: none; border-top: 1px solid #292524; margin: 2rem 0; }
.prose details { background: #1c1917; border: 1px solid #292524; border-radius: 0.5rem; padding: 0.75rem 1rem; margin: 0.75rem 0; }
.prose summary { color: #e7e5e4; cursor: pointer; font-weight: 500; }
.prose summary:hover { color: #fafaf9; }
.prose details[open] summary { margin-bottom: 0.5rem; }

/* Table hover */
.prose tr:hover td { background: #292524; }
</style>
