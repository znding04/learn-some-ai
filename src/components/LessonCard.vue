<template>
  <a href="#" class="lesson-card" @click.prevent="navigate">
    <div class="lesson-card-header">
      <span class="lesson-number">{{ String(index).padStart(2, '0') }}</span>
      <span class="badge" :class="`badge-${lesson.difficulty}`">{{ lesson.difficulty }}</span>
    </div>
    <h3 class="lesson-card-title">{{ lesson.title }}</h3>
    <p class="lesson-card-summary">{{ lesson.summary }}</p>
    <div class="lesson-card-footer">
      <span class="lesson-time">⏱ {{ lesson.estimatedTime }}</span>
      <span class="lesson-arrow">Read →</span>
    </div>
  </a>
</template>

<script setup>
import { useRouter } from '../router.js'

const props = defineProps({
  lesson: Object,
  index: Number,
})

const { navigate: routerNavigate } = useRouter()

function navigate() {
  routerNavigate(`/lesson/${props.lesson.id}`)
}
</script>

<style scoped>
.lesson-card {
  display: block;
  padding: var(--space-5) var(--space-6);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  cursor: pointer;
  transition: box-shadow var(--transition-base), border-color var(--transition-base), transform var(--transition-base);
}

.lesson-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--accent);
  transform: translateY(-2px);
}

.lesson-card-header {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.lesson-number {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-tertiary);
  letter-spacing: 0.05em;
}

.lesson-card-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-heading);
  margin-bottom: var(--space-2);
  line-height: 1.35;
}

.lesson-card-summary {
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.55;
  margin: 0 0 var(--space-4);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.lesson-card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.lesson-time {
  font-size: 0.78rem;
  color: var(--text-tertiary);
  font-weight: 500;
}

.lesson-arrow {
  font-size: 0.8rem;
  color: var(--accent);
  font-weight: 500;
  opacity: 0;
  transform: translateX(-4px);
  transition: opacity var(--transition-base), transform var(--transition-base);
}

.lesson-card:hover .lesson-arrow {
  opacity: 1;
  transform: translateX(0);
}
</style>
