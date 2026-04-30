# 学点啥 (What to Learn)
A serious, structured self-learning platform — no gamification, no badges, no streaks. Just real learning outcomes.

Part of the [ljding.app](https://ljding.app) ecosystem:
- 找谁玩 ([hangwith.ljding.app](https://hangwith.ljding.app)) — Friend hangout tracker
- 玩点啥 ([arcade.ljding.app](https://arcade.ljding.app)) — Game arcade
- 学点啥 ([learn.ljding.app](https://learn.ljding.app)) — Self-learning platform (coming soon)

## Key Features
- Mastery-based progression (not time-bound)
- Curated external resources (YouTube, articles, academic papers)
- Zero gamification — focus on deep learning
- Initial topics: High school math, physics, AI/ML fundamentals
- Free, open-source, hosted on Cloudflare Pages

## Tech Stack
- Frontend: Vue 3 + Vite + Tailwind CSS (matches 玩点啥 stack)
- Hosting: Cloudflare Pages (free tier)
- Content: Markdown files with YAML frontmatter (Git-versioned)
- Math rendering: KaTeX
- Progress tracking: localStorage (MVP) → Cloudflare D1 (post-MVP)

## MVP Scope (2-4 weeks)
1. Topic/module/lesson navigation with prerequisite graphs
2. Markdown lesson pages with KaTeX math rendering
3. Embedded YouTube videos + curated external resource links
4. localStorage-based progress tracking with completion checkboxes
5. Responsive design, deployed to learn.ljding.app

## Research
- Full deep research report: [RESEARCH.md](./RESEARCH.md)
- Quick summary: [SUMMARY.md](./SUMMARY.md)

## Status
🚧 MVP in development — follow progress here.
