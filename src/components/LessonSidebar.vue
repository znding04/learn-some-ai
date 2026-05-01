<template>
  <!-- Mobile toggle -->
  <button
    @click="mobileOpen = !mobileOpen"
    class="lg:hidden fixed bottom-4 left-4 z-50 bg-emerald-600 hover:bg-emerald-500 text-white rounded-full p-3 shadow-lg transition-colors"
    aria-label="Toggle sidebar"
  >
    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" :d="mobileOpen ? 'M6 18L18 6M6 6l12 12' : 'M4 6h16M4 12h16M4 18h16'" />
    </svg>
  </button>

  <!-- Backdrop for mobile -->
  <div
    v-if="mobileOpen"
    @click="mobileOpen = false"
    class="lg:hidden fixed inset-0 bg-black/50 z-40"
  />

  <!-- Sidebar -->
  <aside
    class="fixed lg:sticky top-0 left-0 z-40 h-screen w-72 bg-stone-800 border-r border-stone-700 overflow-y-auto transition-transform duration-300"
    :class="mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'"
  >
    <!-- Header -->
    <div class="p-5 border-b border-stone-700">
      <router-link to="/" class="flex items-center gap-3">
        <span class="text-2xl">📚</span>
        <div>
          <h1 class="text-lg font-bold text-stone-50">学点啥</h1>
          <p class="text-xs text-stone-500">What to Learn</p>
        </div>
      </router-link>
    </div>

    <!-- Topic tree -->
    <nav class="p-3">
      <div v-for="topic in topicsWithLessons" :key="topic.id" class="mb-2">
        <!-- Topic header -->
        <button
          @click="toggleTopic(topic.id)"
          class="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-stone-700/50 transition-colors text-left"
          :class="currentTopicId === topic.id ? 'bg-stone-700/70' : ''"
        >
          <span class="text-lg flex-shrink-0">{{ topic.icon }}</span>
          <div class="flex-1 min-w-0">
            <div class="text-sm font-medium text-stone-200 truncate">{{ topic.title }}</div>
            <div class="text-xs text-stone-500">{{ getTopicProgress(topic.id) }}% complete</div>
          </div>
          <svg
            class="w-4 h-4 text-stone-500 flex-shrink-0 transition-transform duration-200"
            :class="expandedTopics[topic.id] ? 'rotate-90' : ''"
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
          </svg>
        </button>

        <!-- Lessons list -->
        <div
          v-if="expandedTopics[topic.id]"
          class="ml-4 pl-3 border-l border-stone-700 mt-1 space-y-0.5"
        >
          <router-link
            v-for="lesson in topic.lessons"
            :key="lesson.id"
            :to="`/lesson/${lesson.id}`"
            @click="mobileOpen = false"
            class="flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors"
            :class="currentLessonId === lesson.id
              ? 'bg-emerald-700/40 text-emerald-300 font-medium'
              : 'text-stone-400 hover:text-stone-200 hover:bg-stone-700/30'"
          >
            <!-- Completion indicator -->
            <span
              class="w-5 h-5 rounded-full flex items-center justify-center text-xs flex-shrink-0"
              :class="isCompleted(lesson.id) ? 'bg-emerald-600 text-white' : 'bg-stone-700 text-stone-500'"
            >
              <span v-if="isCompleted(lesson.id)">✓</span>
              <span v-else>{{ topic.lessons.indexOf(lesson) + 1 }}</span>
            </span>
            <span class="truncate">{{ lesson.title }}</span>
          </router-link>
        </div>
      </div>
    </nav>

    <!-- Footer: Overall progress -->
    <div class="absolute bottom-0 left-0 right-0 p-4 border-t border-stone-700 bg-stone-800">
      <div class="text-xs text-stone-500 mb-2">Overall Progress</div>
      <div class="flex items-center gap-3">
        <div class="flex-1 h-2 bg-stone-700 rounded-full overflow-hidden">
          <div
            class="h-full bg-emerald-500 rounded-full transition-all"
            :style="{ width: overallProgress + '%' }"
          />
        </div>
        <span class="text-sm text-emerald-400 font-medium">{{ overallProgress }}%</span>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const mobileOpen = ref(false)

const currentLessonId = computed(() => route.params.id)
const currentTopicId = computed(() => {
  // Derive from current lesson's topic
  return lessonsData.value[currentLessonId.value]?.topic || null
})

// Load data
const topicsData = ref([])
const lessonsData = ref({})

onMounted(async () => {
  try {
    const [topicsRes, lessonsRes] = await Promise.all([
      fetch('/content/topics.json'),
      fetch('/content/lessons.json')
    ])
    if (topicsRes.ok) topicsData.value = await topicsRes.json()
    if (lessonsRes.ok) lessonsData.value = await lessonsRes.json()
  } catch (e) {
    console.error('Failed to load sidebar data:', e)
  }
})

// Build topics with lessons
const topicsWithLessons = computed(() => {
  return topicsData.value.map(topic => {
    const lessons = Object.values(lessonsData.value).filter(l => l.topic === topic.id)
    return { ...topic, lessons }
  })
})

// Track which topics are expanded
const expandedTopics = reactive({})

// Auto-expand current topic on mount
onMounted(() => {
  if (currentTopicId.value) {
    expandedTopics[currentTopicId.value] = true
  }
  // Expand all topics by default
  topicsData.value.forEach(t => {
    if (expandedTopics[t.id] === undefined) {
      expandedTopics[t.id] = true
    }
  })
})

function toggleTopic(topicId) {
  expandedTopics[topicId] = !expandedTopics[topicId]
}

function isCompleted(lessonId) {
  try {
    const completed = JSON.parse(localStorage.getItem('wtl-completed') || '[]')
    return completed.includes(lessonId)
  } catch {
    return false
  }
}

function getTopicProgress(topicId) {
  const lessons = Object.values(lessonsData.value).filter(l => l.topic === topicId)
  if (lessons.length === 0) return 0
  const completed = lessons.filter(l => isCompleted(l.id)).length
  return Math.round((completed / lessons.length) * 100)
}

const overallProgress = computed(() => {
  const allLessons = Object.values(lessonsData.value)
  if (allLessons.length === 0) return 0
  const completed = allLessons.filter(l => isCompleted(l.id)).length
  return Math.round((completed / allLessons.length) * 100)
})
</script>
