# 学点啥 — Build Progress

## Current Phase
**Phase 1: Project Scaffolding** — in progress

## Completed Items

### Phase 1: Project Scaffolding
- [x] Scaffold Vue 3 + Vite + Tailwind CSS project (matching 玩点啥 stack)
- [x] Install dependencies: vue-router@4, tailwindcss@3, postcss, autoprefixer, markdown-it, katex
- [x] Configure Tailwind (tailwind.config.js, postcss.config.js)
- [x] Setup Vue Router with placeholder routes: `/`, `/topics`, `/topic/:id`, `/lesson/:id`
- [x] Create directory structure: src/components/, views/, router/, content/, assets/, public/
- [x] Create Home view with overall progress and topic cards
- [x] Create Topics listing view
- [x] Create TopicDetail view with lesson list and progress
- [x] Create LessonView with markdown rendering, KaTeX math, resource links, mark-complete
- [x] Verify build succeeds (`npm run build` → ✓ in 431ms)
- [ ] Initialize Cloudflare Pages (requires interactive auth — deferred to user)

## Next Items

### Phase 2: Core Features (Next Priority)
1. **Lesson View** — connect to real markdown files in `content/` folder instead of inline content
2. **Sidebar navigation** — topic tree sidebar on lesson pages
3. **Breadcrumb navigation** — already in LessonView, verify it works
4. **Progress persistence** — localStorage integration (partially done)
5. **Responsive design** — verify mobile layout

### Phase 3: Content Creation
1. Create `content/topics.json` with full topic metadata
2. Create `content/math-algebra/lessons.json` with lesson registry
3. Write actual markdown lesson files (variables, equations, etc.)
4. Add YouTube video embeds from markdown frontmatter

### Phase 4: Deploy & Polish
1. Connect GitHub repo to Cloudflare Pages (user must do interactive auth)
2. Configure build: `npm run build`, output `dist/`
3. Set custom domain: learn.ljding.app
4. Search functionality, dark mode toggle

## Blockers
- **Cloudflare Pages init**: `npx wrangler pages project create` requires browser OAuth. User needs to run this manually or set up via dashboard.
- **Content**: Need real markdown lesson files. Currently using inline content placeholders.

## Notes
- Stack: Vue 3 + Vite + Tailwind CSS (matching 玩点啥)
- Cloudflare Pages: uses `@cloudflare/vite-plugin` which generates `dist/wrangler.json`
- For Cloudflare Pages: set build command `npm run build`, output dir `dist`
- KaTeX for math rendering (inline `$...$` and block `$$...$$`)
- localStorage key: `wtl-completed` for completed lesson IDs
