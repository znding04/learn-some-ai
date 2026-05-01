<template>
  <div class="flex min-h-screen">
    <!-- Sidebar -->
    <LessonSidebar />

    <!-- Main content -->
    <main class="flex-1 min-w-0">
      <div class="max-w-5xl mx-auto px-6 py-12">
        <!-- Loading state -->
        <div v-if="loading" class="flex items-center justify-center py-20">
          <div class="text-stone-400">Loading lesson...</div>
        </div>

        <!-- Error state -->
        <div v-else-if="error" class="text-center py-20">
          <p class="text-red-400 mb-4">{{ error }}</p>
          <router-link to="/topics" class="text-emerald-400 hover:text-emerald-300">← Back to Topics</router-link>
        </div>

        <!-- Lesson content -->
        <template v-else-if="lesson">
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
            <router-link :to="`/topic/${lesson.topic}`" class="hover:text-stone-300">{{ topicTitle }}</router-link>
            <span>›</span>
            <span class="text-stone-300">{{ lesson.title }}</span>
          </div>

          <!-- Lesson Header -->
          <header class="mb-10">
            <h1 class="text-4xl font-bold text-stone-50 mb-4">{{ lesson.title }}</h1>
            <div class="flex flex-wrap items-center gap-4 text-sm text-stone-400">
              <span v-if="lesson.estimatedTime" class="flex items-center gap-1">
                ⏱ {{ lesson.estimatedTime }}
              </span>
              <span v-if="lesson.difficulty" class="px-2 py-0.5 rounded text-xs font-medium"
                :class="lesson.difficulty === 'beginner' ? 'bg-emerald-900 text-emerald-300' : lesson.difficulty === 'intermediate' ? 'bg-amber-900 text-amber-300' : 'bg-red-900 text-red-300'"
              >
                {{ lesson.difficulty }}
              </span>
              <span v-if="lesson.prerequisites?.length" class="flex items-center gap-1">
                📋 Prerequisites: {{ lesson.prerequisites.join(', ') }}
              </span>
            </div>
          </header>

          <!-- Embedded video -->
          <div v-if="lesson.videoUrl" class="mb-10">
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
          <section v-if="lesson.resources?.length" class="mb-12">
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
        </template>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import MarkdownIt from 'markdown-it'
import katex from 'katex'
import LessonSidebar from '../components/LessonSidebar.vue'

const route = useRoute()
const lessonId = computed(() => route.params.id)

// State
const loading = ref(true)
const error = ref(null)
const lessonsData = ref({})
const allLessonsList = ref([])

// Load lessons metadata and content
onMounted(async () => {
  try {
    // Fetch lessons registry
    const lessonsRes = await fetch('/content/lessons.json')
    if (!lessonsRes.ok) throw new Error('Failed to load lessons data')
    lessonsData.value = await lessonsRes.json()

    // Build all lessons list for prev/next navigation (ordered)
    allLessonsList.value = Object.values(lessonsData.value)

    // Fetch the markdown content for this lesson
    const lesson = lessonsData.value[lessonId.value]
    if (lesson?.contentPath) {
      const contentRes = await fetch(`/content/${lesson.contentPath}`)
      if (!contentRes.ok) throw new Error('Failed to load lesson content')
      lesson.content = await contentRes.text()
    }

    loading.value = false
  } catch (e) {
    console.error('Failed to load lesson:', e)
    error.value = e.message || 'Failed to load lesson'
    loading.value = false
  }
})

// Computed lesson from registry
const lesson = computed(() => {
  return lessonsData.value[lessonId.value] || null
})

const topicTitle = computed(() => {
  if (!lesson.value) return ''
  if (lesson.value.topic === 'math-algebra') return 'Math — Algebra'
  if (lesson.value.topic === 'physics-mechanics') return 'Physics — Mechanics'
  if (lesson.value.topic === 'ai-ml') return 'AI / ML'
  return lesson.value.topic
})

const backLink = computed(() => lesson.value ? `/topic/${lesson.value.topic}` : '/topics')
const backLabel = computed(() => lesson.value ? topicTitle.value : 'Topics')

// Prev/next navigation - sort lessons by their topic order then by some logical ordering
const lessonIndex = computed(() => {
  return allLessonsList.value.findIndex(l => l.id === lessonId.value)
})
const prevLesson = computed(() => lessonIndex.value > 0 ? allLessonsList.value[lessonIndex.value - 1] : null)
const nextLesson = computed(() => lessonIndex.value < allLessonsList.value.length - 1 ? allLessonsList.value[lessonIndex.value + 1] : null)

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
