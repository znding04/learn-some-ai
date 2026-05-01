<template>
  <div class="min-h-screen bg-stone-900 text-stone-100">
    <!-- Top header bar -->
    <header class="sticky top-0 z-30 bg-stone-900/95 backdrop-blur border-b border-stone-800">
      <div class="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
        <router-link to="/" class="flex items-center gap-2 text-stone-100 hover:text-emerald-400 transition-colors">
          <span class="text-xl">📚</span>
          <span class="font-semibold">学点啥</span>
          <span class="text-stone-500 text-sm hidden sm:inline">What to Learn</span>
        </router-link>
        <nav class="flex items-center gap-4 text-sm">
          <router-link to="/" class="text-stone-400 hover:text-stone-100 transition-colors">Home</router-link>
          <router-link to="/topics" class="text-stone-400 hover:text-stone-100 transition-colors">Topics</router-link>
        </nav>
      </div>
    </header>

    <router-view />
  </div>
</template>

<script setup>
</script>
