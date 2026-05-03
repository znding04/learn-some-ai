// Simple hash-based router
import { ref, computed } from 'vue'

const currentRoute = ref(window.location.hash.slice(1) || '/')
const routes = {
  '/': 'home',
  '/topics': 'topics',
}

// Listen for hash changes
window.addEventListener('hashchange', () => {
  currentRoute.value = window.location.hash.slice(1) || '/'
})

export function useRouter() {
  const route = computed(() => currentRoute.value)

  const path = computed(() => {
    const hash = currentRoute.value
    if (!hash || hash === '/') return { name: 'home', params: {} }
    if (hash.startsWith('/topic/')) {
      return { name: 'topic', params: { topicId: hash.slice(7) } }
    }
    if (hash.startsWith('/lesson/')) {
      return { name: 'lesson', params: { lessonId: hash.slice(8) } }
    }
    return { name: 'home', params: {} }
  })

  function navigate(path) {
    window.location.hash = path
  }

  return { route, path, navigate }
}
