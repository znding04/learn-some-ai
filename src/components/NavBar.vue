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
        <a
          href="https://github.com/znding04/learn-some-ai"
          target="_blank"
          rel="noopener noreferrer"
          class="nav-link nav-icon-link"
          aria-label="View source on GitHub"
        >
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.11.79-.25.79-.56v-1.96c-3.2.7-3.88-1.54-3.88-1.54-.52-1.34-1.28-1.7-1.28-1.7-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.62 1.59.23 2.76.11 3.05.74.81 1.18 1.84 1.18 3.1 0 4.42-2.69 5.4-5.25 5.68.41.36.78 1.06.78 2.13v3.16c0 .31.21.68.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/>
          </svg>
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

.nav-icon-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px;
}
</style>
