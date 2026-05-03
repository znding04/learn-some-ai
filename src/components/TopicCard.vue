<template>
  <a href="#" class="topic-card card" @click.prevent="navigate">
    <div class="topic-card-icon">{{ icon }}</div>
    <div class="topic-card-body">
      <h3 class="topic-card-title">{{ title }}</h3>
      <p class="topic-card-desc">{{ description }}</p>
    </div>
    <div class="topic-card-meta">
      <span class="topic-card-count">{{ lessonCount }} lessons</span>
      <span class="topic-card-arrow">→</span>
    </div>
  </a>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from '../router.js'

const props = defineProps({
  id: String,
  title: String,
  icon: String,
  description: String,
  path: String,
  lessonCount: Number,
})

const { navigate: routerNavigate } = useRouter()

function navigate() {
  routerNavigate(`/topic/${props.id}`)
}
</script>

<style scoped>
.topic-card {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  padding: var(--space-5) var(--space-6);
  cursor: pointer;
}

.topic-card-icon {
  font-size: 2rem;
  flex-shrink: 0;
  width: 52px;
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-bg);
  border-radius: var(--radius-md);
  font-size: 1.5rem;
}

.topic-card-body {
  flex: 1;
  min-width: 0;
}

.topic-card-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text-heading);
  margin-bottom: 4px;
}

.topic-card-desc {
  font-size: 0.875rem;
  color: var(--text-secondary);
  line-height: 1.5;
  margin: 0;
}

.topic-card-meta {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex-shrink: 0;
}

.topic-card-count {
  font-size: 0.8rem;
  color: var(--text-tertiary);
  font-weight: 500;
}

.topic-card-arrow {
  font-size: 1rem;
  color: var(--text-tertiary);
  transition: transform var(--transition-base), color var(--transition-base);
}

.topic-card:hover .topic-card-arrow {
  transform: translateX(4px);
  color: var(--accent);
}

.topic-card:hover {
  transform: translateY(-2px);
}
</style>
