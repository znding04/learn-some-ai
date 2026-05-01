<template>
  <div class="max-w-5xl mx-auto px-6 py-12">
    <!-- Loading state -->
    <div v-if="loading" class="flex items-center justify-center py-20">
      <div class="text-stone-400">Loading...</div>
    </div>

    <template v-else-if="topic">
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
    </template>

    <!-- Topic not found -->
    <div v-else class="text-center py-20">
      <p class="text-stone-400 mb-4">Topic not found</p>
      <router-link to="/topics" class="text-emerald-400 hover:text-emerald-300">← Back to Topics</router-link>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const topicId = computed(() => route.params.id)

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
    
    loading.value = false
  } catch (e) {
    console.error('Failed to load data:', e)
    loading.value = false
  }
})

const topic = computed(() => {
  return topicsData.value.find(t => t.id === topicId.value) || null
})

const lessons = computed(() => {
  if (!topic.value) return []
  // Filter lessons that belong to this topic
  return Object.values(lessonsData.value).filter(l => l.topic === topicId.value)
})

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
