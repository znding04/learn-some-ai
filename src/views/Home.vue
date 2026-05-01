<template>
  <div class="max-w-4xl mx-auto px-6 py-16">
    <!-- Header -->
    <header class="text-center mb-16 mt-8">
      <p class="text-stone-500 max-w-lg mx-auto">
        A serious, structured self-learning platform. No gamification, no badges — just mastery-based progression through curated resources.
      </p>
    </header>

    <!-- Topics Preview -->
    <section class="mb-16">
      <h2 class="text-3xl font-semibold text-stone-200 mb-8 text-center">Available Topics</h2>
      <div class="grid gap-6 md:grid-cols-3">
        <router-link
          v-for="topic in topics"
          :key="topic.id"
          :to="`/topic/${topic.id}`"
          class="block bg-stone-800 border border-stone-700 rounded-2xl p-6 hover:border-stone-500 transition-colors group"
        >
          <div class="text-4xl mb-3">{{ topic.icon }}</div>
          <h3 class="text-xl font-semibold text-stone-100 mb-2 group-hover:text-white">{{ topic.title }}</h3>
          <p class="text-stone-400 text-sm mb-4">{{ topic.description }}</p>
          <div class="flex items-center justify-between">
            <span class="text-xs text-stone-500">{{ topic.lessonCount }} lessons</span>
            <span
              class="text-xs font-medium px-2 py-1 rounded"
              :class="topic.difficulty === 'beginner' ? 'bg-emerald-900 text-emerald-300' : topic.difficulty === 'intermediate' ? 'bg-amber-900 text-amber-300' : 'bg-red-900 text-red-300'"
            >
              {{ topic.difficulty }}
            </span>
          </div>
          <!-- Progress bar -->
          <div class="mt-4 h-1 bg-stone-700 rounded-full overflow-hidden">
            <div
              class="h-full bg-emerald-500 rounded-full transition-all"
              :style="{ width: getTopicProgress(topic.id) + '%' }"
            />
          </div>
          <p class="text-xs text-stone-500 mt-2">{{ getTopicProgress(topic.id) }}% complete</p>
        </router-link>
      </div>
    </section>

    <!-- Overall Progress -->
    <section class="bg-stone-800 border border-stone-700 rounded-2xl p-8 mb-16">
      <h2 class="text-2xl font-semibold text-stone-200 mb-4">Your Progress</h2>
      <div class="flex items-center gap-4 mb-4">
        <div class="flex-1 h-3 bg-stone-700 rounded-full overflow-hidden">
          <div
            class="h-full bg-emerald-500 rounded-full transition-all"
            :style="{ width: overallProgress + '%' }"
          />
        </div>
        <span class="text-lg font-medium text-emerald-400">{{ overallProgress }}%</span>
      </div>
      <p class="text-stone-400 text-sm">
        {{ completedLessons }} of {{ totalLessons }} lessons completed across all topics.
      </p>
      <router-link
        to="/topics"
        class="inline-flex items-center gap-2 mt-6 text-emerald-400 hover:text-emerald-300 transition-colors"
      >
        View all topics
        <span>→</span>
      </router-link>
    </section>

    <!-- Principles -->
    <section class="grid gap-6 md:grid-cols-3 mb-16">
      <div class="bg-stone-800 border border-stone-700 rounded-2xl p-6">
        <div class="text-3xl mb-3">🎯</div>
        <h3 class="text-lg font-semibold text-stone-200 mb-2">Mastery-Based</h3>
        <p class="text-stone-400 text-sm">No time limits. Progress when you truly understand the material.</p>
      </div>
      <div class="bg-stone-800 border border-stone-700 rounded-2xl p-6">
        <div class="text-3xl mb-3">📚</div>
        <h3 class="text-lg font-semibold text-stone-200 mb-2">Curated Resources</h3>
        <p class="text-stone-400 text-sm">Quality YouTube videos, articles, and papers — not random internet links.</p>
      </div>
      <div class="bg-stone-800 border border-stone-700 rounded-2xl p-6">
        <div class="text-3xl mb-3">🔔</div>
        <h3 class="text-lg font-semibold text-stone-200 mb-2">Zero Distractions</h3>
        <p class="text-stone-400 text-sm">No badges, no streaks, no leaderboards. Just you and the material.</p>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const topics = [
  {
    id: 'math-algebra',
    title: 'High School Math',
    icon: '📐',
    description: 'Algebra, geometry, trigonometry, and calculus fundamentals.',
    lessonCount: 15,
    difficulty: 'beginner',
  },
  {
    id: 'physics-mechanics',
    title: 'Physics',
    icon: '⚡',
    description: 'Mechanics, electromagnetism, thermodynamics, and modern physics.',
    lessonCount: 12,
    difficulty: 'intermediate',
  },
  {
    id: 'ai-ml',
    title: 'AI / ML',
    icon: '🤖',
    description: 'Machine learning fundamentals, neural networks, and practical AI applications.',
    lessonCount: 10,
    difficulty: 'intermediate',
  },
]

const completedLessons = computed(() => {
  try {
    return JSON.parse(localStorage.getItem('wtl-completed') || '[]').length
  } catch {
    return 0
  }
})

const totalLessons = computed(() => topics.reduce((sum, t) => sum + t.lessonCount, 0))

const overallProgress = computed(() => {
  if (totalLessons.value === 0) return 0
  return Math.round((completedLessons.value / totalLessons.value) * 100)
})

function getTopicProgress(topicId) {
  try {
    const completed = JSON.parse(localStorage.getItem('wtl-completed') || '[]')
    const topic = topics.find(t => t.id === topicId)
    if (!topic) return 0
    // Placeholder: count completed lessons that start with topic prefix
    const topicLessonsCompleted = completed.filter(id => id.startsWith(topicId.split('-')[0]))
    return Math.round((topicLessonsCompleted.length / topic.lessonCount) * 100)
  } catch {
    return 0
  }
}
</script>
