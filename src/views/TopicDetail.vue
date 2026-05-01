<template>
  <div class="max-w-5xl mx-auto px-6 py-12">
    <header class="mb-12">
      <router-link to="/topics" class="text-stone-400 hover:text-stone-200 transition-colors text-sm mb-4 inline-flex items-center gap-1">
        ← Topics
      </router-link>
      <div class="flex items-center gap-4 mb-2">
        <span class="text-5xl">{{ topic.icon }}</span>
        <div>
          <h1 class="text-4xl font-bold text-stone-50">{{ topic.title }}</h1>
          <p class="text-stone-400 mt-1">{{ topic.description }}</p>
        </div>
      </div>
      <!-- Overall progress -->
      <div class="mt-6 flex items-center gap-4">
        <div class="flex-1 h-2 bg-stone-700 rounded-full overflow-hidden">
          <div
            class="h-full bg-emerald-500 rounded-full transition-all"
            :style="{ width: topicProgress + '%' }"
          />
        </div>
        <span class="text-sm text-emerald-400 font-medium">{{ topicProgress }}% complete</span>
      </div>
    </header>

    <!-- Lessons list -->
    <section>
      <h2 class="text-2xl font-semibold text-stone-200 mb-6">Lessons</h2>
      <div class="space-y-3">
        <router-link
          v-for="(lesson, index) in lessons"
          :key="lesson.id"
          :to="`/lesson/${lesson.id}`"
          class="flex items-center gap-4 bg-stone-800 border border-stone-700 rounded-xl p-5 hover:border-stone-500 transition-colors"
        >
          <!-- Completion status -->
          <div
            class="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold"
            :class="isCompleted(lesson.id) ? 'bg-emerald-600 text-white' : 'bg-stone-700 text-stone-400'"
          >
            <span v-if="isCompleted(lesson.id)">✓</span>
            <span v-else>{{ index + 1 }}</span>
          </div>
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-3 mb-1">
              <h3 class="text-lg font-medium text-stone-100">{{ lesson.title }}</h3>
              <span class="text-xs text-stone-500">~{{ lesson.estimatedTime }}</span>
            </div>
            <p class="text-stone-400 text-sm truncate">{{ lesson.summary }}</p>
          </div>
          <div class="flex-shrink-0 text-stone-500">→</div>
        </router-link>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const topicId = computed(() => route.params.id)

const topicMap = {
  'math-algebra': {
    id: 'math-algebra',
    title: 'High School Math — Algebra',
    icon: '📐',
    description: 'The foundation of all mathematics — from variables to polynomials.',
    lessons: [
      { id: 'algebra-variables', title: 'Variables and Expressions', estimatedTime: '25min', summary: 'Understanding what variables are and how to use them in expressions.' },
      { id: 'algebra-equations', title: 'Linear Equations', estimatedTime: '30min', summary: 'Solving equations of the form ax + b = c.' },
      { id: 'algebra-inequalities', title: 'Inequalities', estimatedTime: '25min', summary: 'Working with <, >, ≤, ≥ and representing solutions on a number line.' },
      { id: 'algebra-systems', title: 'Systems of Equations', estimatedTime: '35min', summary: 'Solving two or more equations simultaneously.' },
      { id: 'algebra-polynomials', title: 'Polynomials', estimatedTime: '30min', summary: 'Adding, subtracting, and multiplying polynomial expressions.' },
      { id: 'algebra-factoring', title: 'Factoring', estimatedTime: '40min', summary: 'Factoring quadratics and other polynomial forms.' },
      { id: 'algebra-quadratic', title: 'Quadratic Equations', estimatedTime: '35min', summary: 'Solving quadratic equations using factoring, completing the square, and the quadratic formula.' },
      { id: 'algebra-rational', title: 'Rational Expressions', estimatedTime: '30min', summary: 'Simplifying and operating on rational expressions.' },
      { id: 'algebra-radicals', title: 'Radicals and Exponents', estimatedTime: '35min', summary: 'Working with square roots, nth roots, and exponent rules.' },
      { id: 'algebra-functions', title: 'Introduction to Functions', estimatedTime: '30min', summary: 'What functions are, function notation, and basic function types.' },
      { id: 'algebra-graphing', title: 'Graphing Functions', estimatedTime: '35min', summary: 'Plotting linear and quadratic functions on the coordinate plane.' },
      { id: 'algebra-exponential', title: 'Exponential Functions', estimatedTime: '30min', summary: 'Growth, decay, and the properties of exponential functions.' },
      { id: 'algebra-logarithms', title: 'Logarithms', estimatedTime: '35min', summary: 'The inverse of exponentials — what logarithms are and how to use them.' },
      { id: 'algebra-sequences', title: 'Sequences and Series', estimatedTime: '30min', summary: 'Arithmetic and geometric sequences and summing series.' },
      { id: 'algebra-binomial', title: 'Binomial Theorem', estimatedTime: '30min', summary: 'Expanding (a+b)^n using binomial coefficients.' },
    ],
  },
  'physics-mechanics': {
    id: 'physics-mechanics',
    title: 'Physics — Mechanics',
    icon: '⚡',
    description: 'The laws of motion and the forces that shape our universe.',
    lessons: [
      { id: 'physics-motion', title: 'Motion and Kinematics', estimatedTime: '40min', summary: 'Position, velocity, acceleration, and the equations of motion.' },
      { id: 'physics-vectors', title: 'Vectors in Physics', estimatedTime: '30min', summary: 'Adding, subtracting, and using vectors to describe physical quantities.' },
      { id: 'physics-newton', title: 'Newton\'s Laws of Motion', estimatedTime: '45min', summary: 'The three fundamental laws governing all motion.' },
      { id: 'physics-friction', title: 'Friction and Circular Motion', estimatedTime: '35min', summary: 'Friction forces and the dynamics of circular motion.' },
      { id: 'physics-work', title: 'Work and Energy', estimatedTime: '40min', summary: 'Work, kinetic energy, potential energy, and the work-energy theorem.' },
      { id: 'physics-conservation', title: 'Conservation of Energy', estimatedTime: '35min', summary: 'How energy transforms but is never created or destroyed.' },
      { id: 'physics-momentum', title: 'Momentum and Impulse', estimatedTime: '35min', summary: 'Linear momentum, impulse, and conservation of momentum.' },
      { id: 'physics-collisions', title: 'Collisions', estimatedTime: '35min', summary: 'Elastic and inelastic collisions in one and two dimensions.' },
      { id: 'physics-gravity', title: 'Gravitation', estimatedTime: '40min', summary: 'Newton\'s law of universal gravitation and orbital motion.' },
      { id: 'physics-rotation', title: 'Rotational Motion', estimatedTime: '40min', summary: 'Angular velocity, acceleration, torque, and rotational kinetic energy.' },
      { id: 'physics-oscillations', title: 'Oscillations', estimatedTime: '35min', summary: 'Simple harmonic motion, pendulums, and damped oscillations.' },
      { id: 'physics-waves', title: 'Waves', estimatedTime: '40min', summary: 'Wave properties, interference, and the wave equation.' },
    ],
  },
  'ai-ml': {
    id: 'ai-ml',
    title: 'AI / ML Fundamentals',
    icon: '🤖',
    description: 'Understanding how machines can learn from data and make predictions.',
    lessons: [
      { id: 'ml-intro', title: 'What is Machine Learning?', estimatedTime: '25min', summary: 'Definitions, types of ML (supervised, unsupervised, reinforcement), and key concepts.' },
      { id: 'ml-supervised', title: 'Supervised Learning', estimatedTime: '40min', summary: 'Classification and regression — training models with labeled data.' },
      { id: 'ml-unsupervised', title: 'Unsupervised Learning', estimatedTime: '35min', summary: 'Clustering, dimensionality reduction, and finding patterns without labels.' },
      { id: 'ml-linear-regression', title: 'Linear Regression', estimatedTime: '40min', summary: 'The simplest ML model — fitting a line to data and understanding loss functions.' },
      { id: 'ml-logistic-regression', title: 'Logistic Regression', estimatedTime: '35min', summary: 'Classification using the logistic function and decision boundaries.' },
      { id: 'ml-neural-networks', title: 'Neural Networks', estimatedTime: '50min', summary: 'Perceptrons, layers, activation functions, and backpropagation.' },
      { id: 'ml-deep-learning', title: 'Deep Learning', estimatedTime: '45min', summary: 'Why depth matters — CNNs, regularization, and practical architectures.' },
      { id: 'ml-evaluation', title: 'Model Evaluation', estimatedTime: '35min', summary: 'Train/test splits, cross-validation, precision, recall, F1, and overfitting.' },
      { id: 'ml-decision-trees', title: 'Decision Trees and Ensembles', estimatedTime: '40min', summary: 'Tree-based models, random forests, and gradient boosting.' },
      { id: 'ml-ethics', title: 'AI Ethics and Responsible ML', estimatedTime: '30min', summary: 'Bias, fairness, interpretability, and the social impact of ML systems.' },
    ],
  },
}

const topic = computed(() => topicMap[topicId.value] || topicMap['math-algebra'])
const lessons = computed(() => topic.value.lessons)

const topicProgress = computed(() => {
  try {
    const completed = JSON.parse(localStorage.getItem('wtl-completed') || '[]')
    const total = lessons.value.length
    if (total === 0) return 0
    const done = lessons.value.filter(l => completed.includes(l.id)).length
    return Math.round((done / total) * 100)
  } catch {
    return 0
  }
})

function isCompleted(lessonId) {
  try {
    const completed = JSON.parse(localStorage.getItem('wtl-completed') || '[]')
    return completed.includes(lessonId)
  } catch {
    return false
  }
}
</script>
