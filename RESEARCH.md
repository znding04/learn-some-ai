# 学点啥 (Xue Dian Sha) — Deep Research Report

## Executive Summary

学点啥 is a serious, structured self-learning platform that fills a genuine market gap: no existing platform combines mastery-based progression, curated external resources, and a no-gamification philosophy in a lightweight, self-hostable package. The technical stack (Vue 3 + Vite + Cloudflare Pages + D1) is proven and aligns with the existing ljding.app ecosystem. The MVP is buildable in 2-4 weeks, with content creation being the primary bottleneck — not engineering. The biggest risk is sustained content authoring, which can be mitigated by starting with curated links to existing high-quality resources (Khan Academy videos, 3Blue1Brown, etc.) rather than creating original content.

---

## 1. Competitive Landscape Analysis

### Platform Breakdown

| Platform | Approach | Strengths | Weaknesses for Serious Learners |
|----------|----------|-----------|--------------------------------|
| **Duolingo** | Gamified, casual | Habit formation, accessibility | Shallow depth, gamification over substance, limited STEM |
| **Khan Academy** | Free, video-driven | Comprehensive K-12, excellent math/science | Passive video format, linear progression, no spaced repetition |
| **Brilliant** | Interactive problem-solving | Deep STEM engagement, active learning | Paid ($25/mo), limited subject range, no external resources |
| **Coursera** | University-style courses | Credentialed, deep content | Expensive ($49/mo+), time-bound cohorts, heavy commitment |
| **edX / Open edX** | University MOOCs | Open-source platform, rigorous | Complex to self-host, enterprise-focused, overkill for personal use |
| **MasterClass** | Celebrity-taught | High production value | Entertainment over education, no assessments |
| **Udacity** | Nanodegrees, career-focused | Project-based, industry partnerships | Expensive, narrow tech focus, career-oriented not learning-oriented |

### Market Gap Identified

No existing platform provides ALL of:
1. **Structured mastery-based progression** (not time-based)
2. **Curated external resource aggregation** (leveraging existing great content)
3. **No gamification** (no streaks, badges, leaderboards)
4. **Lightweight and self-hostable** (not enterprise LMS)
5. **Free and open-source**

Khan Academy comes closest but lacks resource curation and spaced repetition. Brilliant has great pedagogy but is paid and closed. Open edX is open-source but massively over-engineered for personal/small-scale use.

---

## 2. Pedagogical Approach

### Mastery-Based vs. Time-Based Learning

**Mastery-based learning** (recommended) requires demonstrating understanding before advancing. Research consistently shows it produces better outcomes than time-based progression:

- Learners move at their own pace
- No "passing with a C" — either you understand it or you don't
- Reduces knowledge gaps that compound over time
- Particularly effective for hierarchical subjects (math, physics)

### Proposed Learning Structure

```
Subject (e.g., High School Math)
  └── Module (e.g., Algebra)
       └── Unit (e.g., Linear Equations)
            └── Lesson (e.g., Solving 2-variable systems)
                 ├── Concept explanation (text + embedded video)
                 ├── External resources (curated links)
                 ├── Practice problems (self-check)
                 └── Checkpoint quiz (mastery verification)
```

### Verification Without Gamification

- **Self-assessment checkboxes**: "I understand this concept" (honor system, suitable for personal use)
- **Checkpoint quizzes**: Multiple-choice or short-answer at unit boundaries
- **Practice problems**: With expandable solutions (not graded, for self-study)
- **Progress indicators**: Simple completion percentages, not XP or levels

### Spaced Repetition Integration

Use **FSRS (Free Spaced Repetition Scheduler)** — the open-source algorithm from the Open Spaced Repetition community:
- Available as a JavaScript/TypeScript library (`ts-fsrs`)
- More effective than SM-2 (Anki's algorithm)
- Can be integrated per-concept for review scheduling
- Implementation: generate review cards from checkpoint questions, schedule reviews based on FSRS intervals

### External Resource Integration

- Embed YouTube videos directly in lessons (iframe embeds, no API key needed for basic embeds)
- Link to articles, papers, textbooks with brief annotations
- Tag resources by difficulty level and format
- Allow community suggestions for resources (full version)

---

## 3. Content Structure Proposal

### Knowledge Graph Approach

Use a **prerequisite graph** (DAG — Directed Acyclic Graph) rather than a simple linear sequence:

```
Arithmetic → Pre-Algebra → Algebra I → Geometry → Algebra II → Precalculus → Calculus
                                ↓                      ↓
                          Trigonometry ────────────────→┘
```

This allows:
- Learners to see what they need to know before starting a topic
- Multiple valid learning paths
- Clear visualization of progress through the graph

### Content Format

Each lesson stored as a **Markdown file** with YAML frontmatter:

```yaml
---
id: algebra-linear-equations-01
title: "Solving Linear Equations"
module: algebra
unit: linear-equations
prerequisites: [pre-algebra-variables-03]
estimated_time: 30min
difficulty: beginner
resources:
  - type: video
    url: https://youtube.com/watch?v=...
    title: "Khan Academy - Linear Equations"
    duration: 12min
  - type: article
    url: https://...
    title: "Paul's Online Math Notes - Linear Equations"
checkpoint:
  - question: "Solve for x: 3x + 7 = 22"
    answer: "x = 5"
    type: short-answer
---

## Introduction

A linear equation is an equation where the highest power of the variable is 1...

## Key Concepts

...

## Practice Problems

1. Solve: 2x + 5 = 13
   <details><summary>Solution</summary>x = 4</details>
```

### Example Progression: High School Math (Algebra → Calculus)

**Module 1: Algebra Foundations** (15 lessons)
- Variables and expressions
- Linear equations (1-variable, 2-variable)
- Inequalities
- Systems of equations
- Polynomials
- Factoring

**Module 2: Functions & Graphs** (12 lessons)
- Function notation
- Linear functions & graphing
- Quadratic functions
- Exponential functions
- Logarithms

**Module 3: Trigonometry** (10 lessons)
- Unit circle
- Trig functions
- Identities
- Applications

**Module 4: Precalculus** (12 lessons)
- Limits (intuitive)
- Sequences and series
- Conic sections
- Polar coordinates

**Module 5: Calculus I** (15 lessons)
- Limits (formal)
- Derivatives
- Applications of derivatives
- Integrals
- Fundamental theorem of calculus

### Resource Curation Strategy

For each lesson, curate 2-4 external resources:
1. **Primary video**: Best explanation (Khan Academy, 3Blue1Brown, Professor Leonard)
2. **Alternative video**: Different teaching style
3. **Text resource**: Written explanation with examples
4. **Practice**: External problem sets (if available)

---

## 4. Technical Feasibility & Stack

### Recommended Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Frontend** | Vue 3 + Vite | Matches 玩点啥 stack, excellent DX, fast builds |
| **Styling** | Tailwind CSS | Consistent with ecosystem, utility-first |
| **Deployment** | Cloudflare Pages | Free tier, edge deployment, matches existing apps |
| **Backend** | Cloudflare Workers (Functions) | Serverless, co-located with Pages, free tier |
| **Database** | Cloudflare D1 (SQLite) | Free tier, simple, sufficient for progress tracking |
| **Content** | Markdown files in repo | Easy to author, version-controlled, no CMS needed |
| **Markdown Rendering** | markdown-it or unified/remark | Client-side rendering from fetched .md files |
| **Math Rendering** | KaTeX | Fast, lightweight LaTeX rendering (critical for math content) |
| **Video Embeds** | YouTube iframe embeds | No API key needed, free, reliable |
| **Spaced Repetition** | ts-fsrs | Open-source FSRS implementation in TypeScript |

### Why Vue 3 + Vite over Next.js

- **Ecosystem consistency**: Matches 玩点啥, shared component library potential
- **Simpler mental model**: No RSC complexity, straightforward SPA
- **Cloudflare-native**: Vite plugin for Cloudflare Workers is v1.0 stable
- **Lighter weight**: Learning platform doesn't need Next.js's SSR complexity
- **If SSR needed later**: Nuxt 4 is the Vue equivalent, easy migration

### Why D1 over PostgreSQL

- **Free tier**: 5M reads/day, 100K writes/day — more than enough
- **Zero ops**: No database server to manage
- **Co-located**: Same Cloudflare ecosystem, minimal latency
- **SQLite**: Simple schema, easy to reason about
- **Migration path**: Can move to Turso (distributed SQLite) if needed

### Content Management: Markdown Files in Git

- Authors write lessons in Markdown with YAML frontmatter
- Content lives in the same repo (or a separate content repo)
- Build step processes Markdown into JSON for the frontend
- Version control provides history, review, and collaboration
- No CMS to maintain, no database for content
- Feels like writing in Obsidian

### Progress Tracking Strategy

**MVP**: localStorage (no auth needed, works immediately)
**Full version**: D1 database with user accounts (Better Auth)

Schema for D1:
```sql
CREATE TABLE user_progress (
  user_id TEXT,
  lesson_id TEXT,
  status TEXT CHECK(status IN ('not_started', 'in_progress', 'completed')),
  completed_at DATETIME,
  PRIMARY KEY (user_id, lesson_id)
);

CREATE TABLE review_cards (
  user_id TEXT,
  card_id TEXT,
  lesson_id TEXT,
  due_date DATETIME,
  stability REAL,
  difficulty REAL,
  reps INTEGER,
  PRIMARY KEY (user_id, card_id)
);
```

---

## 5. Feature Prioritization

### P0 — MVP (Must Have for Launch)

- [ ] Topic/module/lesson navigation with prerequisite indicators
- [ ] Lesson pages rendering Markdown content with KaTeX math
- [ ] Embedded YouTube videos in lessons
- [ ] External resource links with annotations
- [ ] Progress tracking via localStorage
- [ ] Simple completion marking (checkbox per lesson)
- [ ] Responsive design (mobile-friendly)
- [ ] Deploy on Cloudflare Pages

### P1 — Enhanced (Next Iteration)

- [ ] Checkpoint quizzes with auto-checking (multiple choice)
- [ ] Prerequisite graph visualization (D3.js or similar)
- [ ] D1-backed progress tracking
- [ ] User accounts via Better Auth
- [ ] Practice problems with expandable solutions
- [ ] Search across lessons and resources
- [ ] Dark mode

### P2 — Full Version (Future)

- [ ] Spaced repetition system (FSRS-based review scheduling)
- [ ] Short-answer quiz grading
- [ ] Cross-app integration with 找谁玩
- [ ] Community resource suggestions
- [ ] Content contribution workflow (PRs for new lessons)
- [ ] Export/import progress
- [ ] Certificate generation (PDF)
- [ ] Discussion/Q&A per lesson

---

## 6. Suggested Content Structure for First 3 Topics

### Topic 1: High School Math
```
high-school-math/
  ├── _meta.yaml          # Topic metadata, description
  ├── 01-algebra/
  │   ├── _meta.yaml      # Module metadata, prerequisites
  │   ├── 01-variables.md
  │   ├── 02-linear-equations.md
  │   ├── 03-inequalities.md
  │   └── ...
  ├── 02-functions/
  ├── 03-trigonometry/
  ├── 04-precalculus/
  └── 05-calculus/
```

### Topic 2: High School Physics
```
high-school-physics/
  ├── 01-mechanics/
  │   ├── 01-motion.md
  │   ├── 02-forces.md
  │   ├── 03-energy.md
  │   └── ...
  ├── 02-waves/
  ├── 03-electricity/
  ├── 04-magnetism/
  └── 05-modern-physics/
```
Prerequisites: Requires algebra and trigonometry from math topic.

### Topic 3: AI/ML Fundamentals
```
ai-ml-fundamentals/
  ├── 01-math-prereqs/
  │   ├── 01-linear-algebra.md
  │   ├── 02-probability.md
  │   └── 03-calculus-review.md
  ├── 02-ml-basics/
  │   ├── 01-what-is-ml.md
  │   ├── 02-supervised-learning.md
  │   ├── 03-linear-regression.md
  │   └── ...
  ├── 03-deep-learning/
  ├── 04-nlp-basics/
  └── 05-practical-projects/
```
Prerequisites: Requires calculus, linear algebra, basic programming.

---

## 7. Risk Assessment

### Technical Risks (Low)

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Cloudflare D1 limitations | Low | Low | localStorage MVP, migrate to Turso if needed |
| KaTeX rendering edge cases | Low | Medium | Test with complex equations early |
| YouTube embed restrictions | Low | Low | Fallback to direct links |

### Content Risks (High)

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| **Content creation burden** | **High** | **High** | Start with curated links, not original content |
| Content quality inconsistency | Medium | Medium | Use established sources (Khan Academy, 3B1B) |
| Keeping resources up-to-date | Medium | Medium | Community contributions, periodic review |
| Copyright concerns with embeds | Low | Low | YouTube embeds are legal; link, don't copy |

### Adoption Risks (Medium)

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Low user motivation without gamification | Medium | Medium | Target intrinsically motivated learners |
| Competition from established platforms | Medium | High | Differentiate on curation + structure |
| Single-maintainer sustainability | Medium | High | Keep scope small, automate where possible |

---

## 8. Feasibility Assessment

### Development Time Estimates

| Phase | Scope | Estimated Time |
|-------|-------|---------------|
| MVP (P0) | Navigation + lesson rendering + localStorage progress | 2-4 weeks |
| Enhanced (P1) | Quizzes + D1 + auth + search | 3-5 weeks additional |
| Full (P2) | Spaced repetition + community + cross-app | 6-10 weeks additional |
| **Content**: 3 topics, ~60 lessons each | Curation + writing | Ongoing, 1-2 lessons/day |

### Complexity Score: 4/10

The technical complexity is low — it's essentially a content-driven static site with optional dynamic features. The real complexity is in content creation and curation.

### Maintenance Overhead: Low

- Cloudflare Pages: zero server maintenance
- D1: managed database, no ops
- Content updates: Markdown files, version-controlled
- Dependencies: minimal (Vue, Vite, KaTeX, markdown-it)

### Build vs. Fork Decision

**Recommendation: Build from scratch.**

Reasons:
- Open edX, Moodle, etc. are massively over-engineered for this use case
- The custom stack (Vue + Cloudflare) is non-standard for LMS platforms
- The specific design philosophy (no gamification, curated resources) doesn't match any existing platform
- The codebase will be small (~2-5K lines for MVP) — simpler to build than to customize
- Maintains consistency with the ljding.app ecosystem

---

## 9. Monetization & Sustainability

### Recommended: Keep it Free

- **Cloudflare free tier** covers hosting costs ($0)
- **No original video production** = no content costs
- Content authoring is the only "cost" (Zane's time)

### If Monetization Needed Later

1. **Donations** (GitHub Sponsors, Buy Me a Coffee) — lowest friction
2. **Premium content** (advanced topics behind paywall) — not recommended initially
3. **Khan Academy model** (non-profit + grants) — if it grows large enough
4. **Consulting/tutoring** — leverage the platform as a portfolio

---

## 10. Integration with ljding.app Ecosystem

### Cross-App Synergies

| App | Integration Opportunity |
|-----|------------------------|
| **找谁玩** (social) | Share learning milestones, find study partners |
| **玩点啥** (entertainment) | Educational game recommendations, "learn the math behind this game" |
| **学点啥** (learning) | Core learning platform |

### Implementation

- Shared navigation bar across apps (consistent UI)
- Shared auth system (Better Auth, when implemented)
- Link between apps: "Learn the physics behind [game]" → lesson link
- Share progress cards on 找谁玩 (optional, user-initiated, not automatic)

### UI/UX Consistency

- Use same Tailwind CSS configuration
- Shared color palette and typography
- Common header/footer component library
- Consistent responsive breakpoints

---

## 11. Next Steps / Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. Initialize Vue 3 + Vite project with Cloudflare Pages config
2. Set up Tailwind CSS and base layout
3. Create Markdown content processing pipeline (build step)
4. Build lesson rendering with KaTeX support
5. Implement YouTube embed component
6. Write 5-10 sample lessons for algebra module

### Phase 2: Navigation & Progress (Week 3-4)
1. Build topic → module → lesson navigation
2. Implement localStorage-based progress tracking
3. Add completion checkboxes and progress indicators
4. Create external resource link components
5. Mobile-responsive design pass
6. Deploy MVP to Cloudflare Pages

### Phase 3: Content & Polish (Week 5-8)
1. Write/curate content for first 3 topics (ongoing)
2. Add checkpoint quizzes (P1)
3. Implement search functionality
4. Add dark mode
5. Cross-app navigation integration

### Phase 4: Enhanced Features (Week 9+)
1. D1 database setup for progress persistence
2. User accounts (Better Auth)
3. Prerequisite graph visualization
4. Spaced repetition integration (FSRS)
5. Community features

---

*Research compiled: April 2026*
*Sources: Khan Academy, Brilliant, Coursera, Open edX, Cloudflare documentation, Open Spaced Repetition project, FSRS research*
