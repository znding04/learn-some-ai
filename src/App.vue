<template>
  <div id="app">
    <NavBar />

    <main class="page-content">
      <!-- Home: Topic List -->
      <div v-if="routeName === 'home'" class="container fade-in">
        <div class="home-hero">
          <h1 class="hero-title">Learn AI, <span class="hero-accent">step by step</span></h1>
          <p class="hero-subtitle">
            A structured curriculum covering AI fundamentals, large language models, and AI agents — from first principles to production.
          </p>
        </div>

        <div class="topics-section">
          <div class="section-label">Curriculum Tracks</div>
          <div class="topics-grid">
            <TopicCard
              v-for="(topic, i) in topics"
              :key="topic.id"
              :id="topic.id"
              :title="topic.title"
              :icon="topic.icon"
              :description="topic.description"
              :path="topic.path"
              :lesson-count="getTopicLessonCount(topic.id)"
              :class="`fade-in-delay-${i + 1}`"
            />
          </div>
        </div>

        <div class="stats-section">
          <div class="stat-item">
            <span class="stat-number">{{ totalLessons }}</span>
            <span class="stat-label">lessons</span>
          </div>
          <div class="stat-divider"></div>
          <div class="stat-item">
            <span class="stat-number">{{ totalTopics }}</span>
            <span class="stat-label">tracks</span>
          </div>
          <div class="stat-divider"></div>
          <div class="stat-item">
            <span class="stat-number">Free</span>
            <span class="stat-label">forever</span>
          </div>
        </div>
      </div>

      <!-- Topic Detail -->
      <div v-else-if="routeName === 'topic'" class="container fade-in">
        <div class="topic-header">
          <a href="#/" class="back-link">← All Topics</a>
          <div class="topic-header-icon">{{ currentTopic?.icon }}</div>
          <h1 class="topic-header-title">{{ currentTopic?.title }}</h1>
          <p class="topic-header-desc">{{ currentTopic?.description }}</p>
          <div class="topic-header-stats">
            <span>{{ getTopicLessonCount(currentTopicId) }} lessons</span>
          </div>
        </div>

        <div class="lessons-grid">
          <LessonCard
            v-for="(lesson, i) in topicLessons"
            :key="lesson.id"
            :lesson="lesson"
            :index="i + 1"
            :class="`fade-in-delay-${Math.min(i + 1, 6)}`"
          />
        </div>
      </div>

      <!-- Lesson View -->
      <div v-else-if="routeName === 'lesson'" class="container">
        <LessonView
          :lesson-id="lessonId"
          :lessons-data="lessons"
          :topics-data="topics"
        />
      </div>
    </main>

    <footer class="site-footer">
      <div class="container footer-inner">
        <span>学点AI — Learn AI step by step</span>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import NavBar from './components/NavBar.vue'
import TopicCard from './components/TopicCard.vue'
import LessonCard from './components/LessonCard.vue'
import LessonView from './components/LessonView.vue'
import { useRouter } from './router.js'

const { route, path } = useRouter()

const lessons = ref({})
const topics = ref([])

const routeName = computed(() => path.value.name)
const topicId = computed(() => path.value.params.topicId)
const lessonId = computed(() => path.value.params.lessonId)

const currentTopic = computed(() =>
  topics.value.find(t => t.id === topicId.value)
)

const topicLessons = computed(() => {
  if (!topicId.value) return []
  return Object.values(lessons.value)
    .filter(l => l.topic === topicId.value)
    .sort((a, b) => {
      const numA = parseInt(a.contentPath.split('/').pop().split('-')[0])
      const numB = parseInt(b.contentPath.split('/').pop().split('-')[0])
      return numA - numB
    })
})

const totalLessons = computed(() => Object.keys(lessons.value).length)
const totalTopics = computed(() => topics.value.length)

function getTopicLessonCount(topicIdVal) {
  return Object.values(lessons.value).filter(l => l.topic === topicIdVal).length
}

onMounted(async () => {
  try {
    const [lessonsRes, topicsRes] = await Promise.all([
      fetch('/content/lessons.json'),
      fetch('/content/topics.json'),
    ])
    lessons.value = await lessonsRes.json()
    topics.value = await topicsRes.json()
  } catch (e) {
    console.error('Failed to load content:', e)
  }
})
</script>

<style scoped>
/* Home Hero */
.home-hero {
  text-align: center;
  padding: var(--space-12) 0 var(--space-10);
}

.hero-title {
  font-size: 2.75rem;
  font-weight: 700;
  color: var(--text-heading);
  letter-spacing: -0.04em;
  line-height: 1.15;
  margin-bottom: var(--space-4);
}

.hero-accent {
  background: linear-gradient(135deg, var(--accent) 0%, #a855f7 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 1.05rem;
  color: var(--text-secondary);
  max-width: 560px;
  margin: 0 auto;
  line-height: 1.65;
}

/* Topics Section */
.topics-section {
  margin-bottom: var(--space-12);
}

.section-label {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-tertiary);
  margin-bottom: var(--space-4);
}

.topics-grid {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

/* Stats Section */
.stats-section {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-6);
  padding: var(--space-6) 0;
  border-top: 1px solid var(--border);
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.stat-number {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-heading);
  letter-spacing: -0.02em;
}

.stat-label {
  font-size: 0.75rem;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat-divider {
  width: 1px;
  height: 32px;
  background: var(--border);
}

/* Topic Header */
.topic-header {
  padding: var(--space-8) 0 var(--space-8);
  border-bottom: 1px solid var(--border);
  margin-bottom: var(--space-8);
}

.back-link {
  display: inline-flex;
  align-items: center;
  font-size: 0.8rem;
  color: var(--text-tertiary);
  margin-bottom: var(--space-5);
  transition: color var(--transition-fast);
}

.back-link:hover {
  color: var(--accent);
}

.topic-header-icon {
  font-size: 2.5rem;
  margin-bottom: var(--space-3);
}

.topic-header-title {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text-heading);
  margin-bottom: var(--space-3);
}

.topic-header-desc {
  font-size: 1rem;
  color: var(--text-secondary);
  line-height: 1.6;
  max-width: 560px;
  margin: 0 0 var(--space-4);
}

.topic-header-stats {
  font-size: 0.85rem;
  color: var(--text-tertiary);
  font-weight: 500;
}

/* Lessons Grid */
.lessons-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: var(--space-4);
}

/* Footer */
.site-footer {
  border-top: 1px solid var(--border);
  padding: var(--space-6) 0;
}

.footer-inner {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  color: var(--text-tertiary);
}

/* Responsive */
@media (max-width: 640px) {
  .hero-title {
    font-size: 2rem;
  }

  .topics-grid {
    gap: var(--space-2);
  }

  .lessons-grid {
    grid-template-columns: 1fr;
  }

  .stats-section {
    gap: var(--space-4);
  }
}
</style>
