# 学点啥 — Build Progress

## Current Phase
**Phase 2: Core Features** — In Progress

## Completed Items

### Phase 1: Project Scaffolding ✅
- [x] Scaffold Vue 3 + Vite + Tailwind CSS project (matching 玩点啥 stack)
- [x] Install dependencies: vue-router@4, tailwindcss@3, postcss, autoprefixer, markdown-it, katex
- [x] Configure Tailwind (tailwind.config.js, postcss.config.js)
- [x] Setup Vue Router with routes: `/`, `/topics`, `/topic/:id`, `/lesson/:id`
- [x] Create directory structure: src/views/, src/components/, src/router/, src/assets/, public/
- [x] Create Home view with overall progress and topic cards
- [x] Create Topics listing view
- [x] Create TopicDetail view with lesson list and progress bars
- [x] Create LessonView: markdown rendering, KaTeX math ($inline$, $$block$$), resource links, prev/next nav, mark-complete
- [x] localStorage progress tracking (wtl-completed key)
- [x] Cloudflare Pages ready: @cloudflare/vite-plugin + wrangler.toml + dist/wrangler.json
- [x] 4 sample lessons with real content (algebra variables, linear equations, physics motion, ML intro)
- [x] Verify build succeeds (`npm run build` → ✓ 462KB JS, 12KB CSS)
- [x] git commit + push to main

### Phase 2: Core Features ✅ (in progress)
- [x] **Lesson View** — connect to real markdown files in `content/` folder instead of inline content
- [x] Create `public/content/topics.json` with topic metadata
- [x] Create `public/content/lessons.json` with lesson registry
- [x] Create actual markdown lesson files in `public/content/` (algebra, physics, ai)
- [x] Update LessonView.vue to fetch markdown content dynamically from `/content/` paths
- [x] Update TopicDetail.vue and Topics.vue to load from JSON registries
- [x] Remove inline lessonRegistry from LessonView.vue (now external content)
- [x] **Sidebar navigation** — topic tree sidebar on lesson pages (LessonSidebar component)
- [x] **Breadcrumb navigation** — verify it works
- [x] **App header** — consistent header bar across all views
- [x] **Responsive layout** — mobile-friendly with sidebar toggle

## Next Items

### Phase 2: Core Features (Next Priority)
1. ~~**Sidebar navigation** — topic tree sidebar on lesson pages~~ ✓ Done
2. ~~**Breadcrumb navigation** — verify it works~~ ✓ Done (already working)
3. ~~**Responsive design** — verify mobile layout~~ ✓ Done (mobile toggle added)
4. Search functionality (Phase 4)
5. Dark mode toggle (Phase 4)

### Phase 3: Content Creation
1. Add more algebra lessons (inequalities, systems, polynomials, factoring, quadratics, etc.)
2. Add more physics lessons to match the topic outline
3. Add more AI/ML lessons to match the topic outline
4. Add YouTube video embeds from markdown frontmatter

### Phase 4: Deploy & Polish
1. Connect GitHub repo to Cloudflare Pages (user must do interactive auth)
2. Configure build: `npm run build`, output dir `dist`
3. Set custom domain: learn.ljding.app
4. Search functionality, dark mode toggle

## Blockers
- **Cloudflare Pages init**: `npx wrangler pages project create` requires browser OAuth. User needs to run this manually or set up via dashboard.

## Notes
- Stack: Vue 3 + Vite + Tailwind CSS (matching 玩点啥)
- Cloudflare Pages: uses `@cloudflare/vite-plugin` which generates `dist/wrangler.json`
- For Cloudflare Pages: set build command `npm run build`, output dir `dist`
- KaTeX for math rendering (inline `$...$` and block `$$...$$`)
- localStorage key: `wtl-completed` for completed lesson IDs
- Content now loaded dynamically from `public/content/` directory
- Lesson metadata in `lessons.json`, topic metadata in `topics.json`
