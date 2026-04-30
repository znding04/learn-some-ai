# 学点啥 (Xue Dian Sha) — Research Summary

## What Is It?

A serious, structured self-learning platform for high school math, physics, and AI/ML — no gamification, no badges, no streaks. Hosted at learn.ljding.app, complementing the existing 找谁玩 and 玩点啥 apps.

## Key Finding: There's a Real Gap

No existing platform combines structured mastery-based progression, curated external resources, and a lightweight self-hostable architecture. Khan Academy is closest but lacks resource curation and spaced repetition. Brilliant has great pedagogy but is paid and closed. Open edX is open-source but wildly over-engineered for personal use.

## Recommended Tech Stack

- **Frontend**: Vue 3 + Vite (matches existing ecosystem)
- **Styling**: Tailwind CSS
- **Hosting**: Cloudflare Pages (free tier)
- **Backend**: Cloudflare Workers (when needed)
- **Database**: Cloudflare D1 (for progress tracking, post-MVP)
- **Content**: Markdown files with YAML frontmatter in Git
- **Math**: KaTeX for LaTeX rendering
- **Spaced Repetition**: ts-fsrs (open-source FSRS algorithm, post-MVP)

Total hosting cost: $0 on Cloudflare free tier.

## MVP Scope (2-4 Weeks)

1. Topic/module/lesson navigation
2. Markdown lesson pages with KaTeX math rendering
3. Embedded YouTube videos and curated resource links
4. localStorage-based progress tracking
5. Simple completion marking
6. Responsive design
7. Deployed on Cloudflare Pages

## Biggest Risk: Content Creation

The platform is technically simple (complexity 4/10). The real challenge is writing and curating ~180 lessons across 3 topics. Mitigation: start as a **resource aggregator** (curated links to Khan Academy, 3Blue1Brown, Professor Leonard, etc.) with light original text, not a content creator. Let the content grow organically.

## Recommendation: Proceed

**Yes, build it.** The MVP is small, the stack is familiar, hosting is free, and the concept fills a genuine gap. Start with the algebra module (15 lessons of curated resources + brief explanations), deploy it, and iterate. The worst case is a useful personal learning dashboard; the best case is a valuable open-source learning platform.

Build from scratch — don't fork an existing LMS. The scope is small enough that custom code will be simpler than fighting an enterprise platform's assumptions.
