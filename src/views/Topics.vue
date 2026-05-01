<template>
  <div class="max-w-5xl mx-auto px-6 py-12">
    <header class="mb-12">
      <router-link to="/" class="text-stone-400 hover:text-stone-200 transition-colors text-sm mb-4 inline-flex items-center gap-1">
        ← Back to home
      </router-link>
      <h1 class="text-4xl font-bold text-stone-50 mb-2">Topics</h1>
      <p class="text-stone-400">Choose a subject area to start learning.</p>
    </header>

    <div class="space-y-6">
      <router-link
        v-for="topic in topics"
        :key="topic.id"
        :to="`/topic/${topic.id}`"
        class="block bg-stone-800 border border-stone-700 rounded-2xl p-8 hover:border-stone-500 transition-colors"
      >
        <div class="flex items-start gap-6">
          <div class="text-5xl">{{ topic.icon }}</div>
          <div class="flex-1">
            <div class="flex items-center justify-between mb-2">
              <h2 class="text-2xl font-semibold text-stone-100">{{ topic.title }}</h2>
              <span
                class="text-xs font-medium px-2 py-1 rounded"
                :class="topic.difficulty === 'beginner' ? 'bg-emerald-900 text-emerald-300' : topic.difficulty === 'intermediate' ? 'bg-amber-900 text-amber-300' : 'bg-red-900 text-red-300'"
              >
                {{ topic.difficulty }}
              </span>
            </div>
            <p class="text-stone-400 mb-4">{{ topic.description }}</p>
            <div class="flex items-center gap-6 text-sm text-stone-500">
              <span>{{ topic.lessonCount }} lessons</span>
              <span>~{{ topic.totalTime }}</span>
              <span class="flex items-center gap-2">
                <span class="text-emerald-400 font-medium">{{ getTopicProgress(topic.id) }}%</span> complete
              </span>
            </div>
            <!-- Progress bar -->
            <div class="mt-4 h-2 bg-stone-700 rounded-full overflow-hidden">
              <div
                class="h-full bg-emerald-500 rounded-full transition-all"
                :style="{ width: getTopicProgress(topic.id) + '%' }"
              />
            </div>
          </div>
        </div>
      </router-link>
    </div>
  </div>
</template>

<script setup>
const topics = [
  {
    id: 'math-algebra',
    title: 'High School Math — Algebra',
    icon: '📐',
    description: 'Variables, equations, inequalities, polynomials, factoring, and more. The foundation of all mathematics.',
    lessonCount: 15,
    totalTime: '6 hours',
    difficulty: 'beginner',
  },
  {
    id: 'physics-mechanics',
    title: 'Physics — Mechanics',
    icon: '⚡',
    description: 'Motion, forces, energy, momentum, and the laws governing the physical universe.',
    lessonCount: 12,
    totalTime: '5 hours',
    difficulty: 'intermediate',
  },
  {
    id: 'ai-ml',
    title: 'AI / ML Fundamentals',
    icon: '🤖',
    description: 'Machine learning basics, neural networks, and how to build intelligent systems.',
    lessonCount: 10,
    totalTime: '8 hours',
    difficulty: 'intermediate',
  },
]

function getTopicProgress(topicId) {
  try {
    const completed = JSON.parse(localStorage.getItem('wtl-completed') || '[]')
    const topic = topics.find(t => t.id === topicId)
    if (!topic) return 0
    const topicLessonsCompleted = completed.filter(id => id.startsWith(topicId.split('-')[0]))
    return Math.round((topicLessonsCompleted.length / topic.lessonCount) * 100)
  } catch {
    return 0
  }
}
</script>
