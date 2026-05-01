import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Topics from '../views/Topics.vue'
import TopicDetail from '../views/TopicDetail.vue'
import LessonView from '../views/LessonView.vue'

const routes = [
  { path: '/', component: Home },
  { path: '/topics', component: Topics },
  { path: '/topic/:id', component: TopicDetail },
  { path: '/lesson/:id', component: LessonView },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
