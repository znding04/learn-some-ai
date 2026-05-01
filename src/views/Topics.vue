<template>
  <div class="max-w-5xl mx-auto px-6 py-12">
    <header class="mb-12">
      <router-link to="/" class="text-stone-400 hover:text-stone-200 transition-colors text-sm mb-4 inline-flex items-center gap-1">
        ← Back to home
      </router-link>
      <h1 class="text-4xl font-bold text-stone-50 mb-2">Topics</h1>
      <p class="text-stone-400">Choose a subject area to start learning.</p>
    </header>

    <!-- Loading state -->
    <div v-if="loading" class="flex items-center justify-center py-20">
      <div class="text-stone-400">Loading topics...</div>
    </div>

    <div v-else class="space-y-6">
      <router-link
        v-for="topic in topicsWithStats"
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
                :class="topic.difficulty || 'bg-stone-700 text-stone-400'"
              >
                {{ topic.difficulty || 'beginner' }}
              </span>
            </div>
            <p class="text-stone-400 mb-4">{{ topic.description }}</p>
            <div class="flex items-center gap-6 text-sm text-stone-500">
              <span>{{ topic.lessonCount }} lessons</span>
              <span class="flex items-center gap-2">
                <span class="text-emerald-400 font-medium">{{ topic.progress }}%</span> complete
              </span>
            </div>
            <!-- Progress bar -->
            <div class="mt-4 h-2 bg-stone-700 rounded-full overflow-hidden">
              <div
                class="h-full bg-emerald-500 rounded-full transition-all"
                :style="{ width: topic.progress + '%' }"
              />
            </div>
          </div>
        </div>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'

// State
const loading = ref(true)
const topicsData = ref([])
const lessonsData = ref({})

// Load data from JSON
onMounted(async () => {
  try {
    const [topicsRes, lessonsRes] = await Promise.all([
      fetch('/content/topics.json'),
      fetch('/content/lessons.json')
    ])
    
    if (topicsRes.ok) {
      topicsData.value = await topicsRes.json()
    }
    
    if (lessonsRes.ok) {
      lessonsData.value = await lessonsRes.json()
    }
  } catch (e) {
    console.error('Failed to load data:', e)
  } finally {
    loading.value = false
  }
})

// Compute topics with lesson counts and progress
const topicsWithStats = computed(() => {
  return topicsData.value.map(topic => {
    const lessons = Object.values(lessonsData.value).filter(l => l.topic === topic.id)
    const completed = getCompleted()
    const lessonCount = lessons.length
    const doneCount = lessons.filter(l => completed.includes(l.id)).length
    const progress = lessonCount > 0 ? Math.round((doneCount / lessonCount) * 100) : 0
    
    return {
      ...topic,
      lessonCount,
      progress,
      difficulty: 'beginner' // Default for now, can be enhanced later
    }
  })
})

function getCompleted() {
  try {
    return JSON.parse(localStorage.getItem('wtl-completed') || '[]')
  } catch {
    return []
  }
}
</script>
