<template>
  <div class="lesson-view">
    <!-- Loading State -->
    <div v-if="loading" class="lesson-loading">
      <div class="loading-skeleton">
        <div class="skeleton-title"></div>
        <div class="skeleton-meta"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line short"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line medium"></div>
      </div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="lesson-error">
      <p>Failed to load lesson content.</p>
      <button @click="loadLesson" class="retry-btn">Retry</button>
    </div>

    <!-- Lesson Content -->
    <article v-else class="lesson-article fade-in">
      <header class="lesson-header">
        <div class="lesson-breadcrumb">
          <a href="#/" class="breadcrumb-link">Topics</a>
          <span class="breadcrumb-sep">/</span>
          <a href="#" @click.prevent="goToTopic" class="breadcrumb-link">{{ topicTitle }}</a>
        </div>
        <div class="lesson-meta">
          <span class="badge" :class="`badge-${lesson.difficulty}`">{{ lesson.difficulty }}</span>
          <span class="lesson-time">⏱ {{ lesson.estimatedTime }}</span>
        </div>
        <h1 class="lesson-title">{{ lesson.title }}</h1>
        <p class="lesson-summary">{{ lesson.summary }}</p>
      </header>

      <div class="lesson-body markdown-body" v-html="renderedContent"></div>

      <footer class="lesson-footer">
        <div class="divider"></div>
        <div class="lesson-nav-footer">
          <a v-if="prevLesson" href="#" @click.prevent="goToLesson(prevLesson.id)" class="nav-btn">
            ← {{ prevLesson.title }}
          </a>
          <span v-else></span>
          <a v-if="nextLesson" href="#" @click.prevent="goToLesson(nextLesson.id)" class="nav-btn">
            {{ nextLesson.title }} →
          </a>
        </div>
      </footer>
    </article>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useRouter } from '../router.js'

const props = defineProps({
  lessonId: String,
  lessonsData: Object,
  topicsData: Array,
})

const router = useRouter()
const loading = ref(false)
const error = ref(false)
const rawContent = ref('')

const lesson = computed(() => props.lessonsData?.[props.lessonId])
const topicTitle = computed(() => {
  const t = props.topicsData?.find(t => t.id === lesson.value?.topic)
  return t?.title || ''
})

// Find prev/next lessons in same topic
const topicLessons = computed(() => {
  if (!lesson.value) return []
  const topic = lesson.value.topic
  return Object.values(props.lessonsData || {})
    .filter(l => l.topic === topic)
    .sort((a, b) => {
      const numA = parseInt(a.contentPath.split('/').pop().split('-')[0])
      const numB = parseInt(b.contentPath.split('/').pop().split('-')[0])
      return numA - numB
    })
})

const currentIndex = computed(() =>
  topicLessons.value.findIndex(l => l.id === props.lessonId)
)
const prevLesson = computed(() =>
  currentIndex.value > 0 ? topicLessons.value[currentIndex.value - 1] : null
)
const nextLesson = computed(() =>
  currentIndex.value < topicLessons.value.length - 1
    ? topicLessons.value[currentIndex.value + 1]
    : null
)

// Simple markdown to HTML converter
function renderMarkdown(md) {
  if (!md) return ''

  let html = md
    // Remove frontmatter
    .replace(/^---[\s\S]*?---\n/, '')
    // Headings
    .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold & Italic
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Code blocks
    .replace(/```[\s\S]*?```/g, (match) => {
      const code = match.replace(/```\w*\n?/, '').replace(/```$/, '')
      return `<pre><code>${code.trim()}</code></pre>`
    })
    // Tables
    .replace(/\|\s*([^\n]+)\|\n\|[-:\s]+\|\n((?:\|[^\n]+\|\n?)*)/g, (match, header, body) => {
      const headers = header.split('|').filter(h => h.trim())
      const rows = body.trim().split('\n').map(row =>
        row.split('|').filter(c => c.trim())
      )
      let table = '<table><thead><tr>'
      headers.forEach(h => { table += `<th>${h.trim()}</th>` })
      table += '</tr></thead><tbody>'
      rows.forEach(row => {
        table += '<tr>'
        row.forEach(c => { table += `<td>${c.trim()}</td>` })
        table += '</tr>'
      })
      table += '</tbody></table>'
      return table
    })
    // Blockquotes
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // Horizontal rule
    .replace(/^---$/gm, '<hr>')
    // Unordered lists
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
    // Ordered lists
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // Wrap consecutive li in ul
    .replace(/(<li>.*<\/li>\n?)+/g, match => `<ul>${match}</ul>`)
    // Paragraphs (lines not yet wrapped)
    .replace(/^(?!<[a-z])(.+)$/gm, '<p>$1</p>')
    // Clean up empty paragraphs
    .replace(/<p>\s*<\/p>/g, '')

  return html
}

const renderedContent = computed(() => renderMarkdown(rawContent.value))

async function loadLesson() {
  if (!lesson.value?.contentPath) return
  loading.value = true
  error.value = false

  try {
    const res = await fetch(`/content/${lesson.value.contentPath}`)
    if (!res.ok) throw new Error('not found')
    rawContent.value = await res.text()
  } catch (e) {
    error.value = true
  } finally {
    loading.value = false
  }
}

function goToTopic() {
  router.navigate(`/topic/${lesson.value.topic}`)
}

function goToLesson(id) {
  router.navigate(`/lesson/${id}`)
}

watch(() => props.lessonId, () => {
  loadLesson()
}, { immediate: true })
</script>

<style scoped>
.lesson-view {
  max-width: 800px;
  margin: 0 auto;
}

.lesson-loading {
  padding: var(--space-8) 0;
}

.loading-skeleton {
  animation: pulse 1.5s ease-in-out infinite;
}

.skeleton-title {
  height: 40px;
  background: var(--bg-tertiary);
  border-radius: var(--radius-sm);
  width: 70%;
  margin-bottom: var(--space-4);
}

.skeleton-meta {
  height: 24px;
  background: var(--bg-tertiary);
  border-radius: var(--radius-sm);
  width: 40%;
  margin-bottom: var(--space-6);
}

.skeleton-line {
  height: 16px;
  background: var(--bg-tertiary);
  border-radius: var(--radius-sm);
  margin-bottom: var(--space-3);
}

.skeleton-line.short { width: 60%; }
.skeleton-line.medium { width: 80%; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.lesson-error {
  text-align: center;
  padding: var(--space-12);
  color: var(--text-secondary);
}

.retry-btn {
  margin-top: var(--space-4);
  padding: 8px 20px;
  background: var(--accent);
  color: white;
  border-radius: var(--radius-sm);
  font-weight: 500;
}

.lesson-article {
  padding: var(--space-8) 0 var(--space-16);
}

.lesson-breadcrumb {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: 0.8rem;
  color: var(--text-tertiary);
  margin-bottom: var(--space-5);
}

.breadcrumb-link {
  color: var(--text-tertiary);
  transition: color var(--transition-fast);
}

.breadcrumb-link:hover {
  color: var(--accent);
}

.breadcrumb-sep {
  opacity: 0.5;
}

.lesson-meta {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.lesson-time {
  font-size: 0.8rem;
  color: var(--text-tertiary);
  font-weight: 500;
}

.lesson-title {
  font-size: 2rem;
  font-weight: 700;
  color: var(--text-heading);
  letter-spacing: -0.03em;
  line-height: 1.2;
  margin-bottom: var(--space-4);
}

.lesson-summary {
  font-size: 1.05rem;
  color: var(--text-secondary);
  line-height: 1.65;
  margin: 0;
}

.lesson-body {
  margin-top: var(--space-8);
}

.lesson-footer {
  margin-top: var(--space-10);
}

.lesson-nav-footer {
  display: flex;
  justify-content: space-between;
  gap: var(--space-4);
}

.nav-btn {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--accent);
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  transition: background var(--transition-fast), border-color var(--transition-fast);
  max-width: 45%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.nav-btn:hover {
  background: var(--accent-bg);
  border-color: var(--accent);
}
</style>
