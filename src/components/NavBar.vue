<template>
  <header class="navbar">
    <div class="container navbar-inner">
      <a href="#/" class="navbar-brand">
        <span class="brand-icon">⚡</span>
        <span class="brand-name">学点AI</span>
      </a>

      <nav class="navbar-nav">
        <a href="#/" class="nav-link" :class="{ active: route === '/' }">
          Topics
        </a>
      </nav>
    </div>
  </header>
</template>

<script setup>
import { useRouter } from '../router.js'
const { route } = useRouter()
</script>

<style scoped>
.navbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

.navbar-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 60px;
}

.navbar-brand {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--text-heading);
  letter-spacing: -0.02em;
  transition: opacity var(--transition-fast);
}

.navbar-brand:hover {
  opacity: 0.75;
}

.brand-icon {
  font-size: 1.2rem;
}

.brand-name {
  font-weight: 700;
}

.navbar-nav {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.nav-link {
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary);
  transition: background var(--transition-fast), color var(--transition-fast);
}

.nav-link:hover {
  background: var(--bg-tertiary);
  color: var(--text-heading);
}

.nav-link.active {
  color: var(--text-heading);
  background: var(--bg-secondary);
}
</style>
