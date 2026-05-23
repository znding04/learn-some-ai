---
title: "NLP for Sports Commentary and Reporting"
level: intermediate
topic: ai-for-sports-science
order: 9
---

# NLP for Sports Commentary and Reporting

## Overview

Sports generate enormous quantities of text: live commentary, match reports, player interviews, social media reactions, statistical summaries, and fantasy sports analysis. Natural Language Processing (NLP) transforms this unstructured text into structured insights, automated content generation, and real-time fan engagement at scale.

This lesson covers text generation for live sports commentary, sentiment analysis for fan reactions, automated match report writing, and the architecture of sports-specific language models.

---

## Live Sports Commentary Generation

### The Commentary Generation Pipeline

Live commentary must be timely, accurate, and engaging. An NLP commentary system processes real-time events and generates natural language descriptions:

```python
import torch
import torch.nn as nn

class CommentaryGenerator(nn.Module):
    """
    Sequence-to-sequence model for generating sports commentary.
    """
    def __init__(self, vocab_size=20000, embed_dim=256, hidden_dim=512):
        super().__init__()

        # Event encoder: encode game state changes
        self.event_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim),  # 64-dim event features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Text decoder with attention
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.LSTM(
            embed_dim + hidden_dim,  # Context + previous token
            hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Attention for event context
        self.event_attention = nn.Linear(hidden_dim, 1)

    def forward(self, event_features, target_text=None, max_len=50):
        """
        Generate commentary conditioned on event features.
        """
        # Encode event
        event_enc = self.event_encoder(event_features)  # (batch, hidden)

        # Decode text
        batch_size = event_features.size(0)
        device = event_features.device

        if target_text is not None:
            # Training mode
            tokens = self.embedding(target_text)  # (batch, seq, embed)
            context = event_enc.unsqueeze(1).expand(-1, tokens.size(1), -1)
            decoder_input = torch.cat([tokens, context], dim=-1)

            outputs, _ = self.decoder(decoder_input)
            logits = self.output_proj(outputs)
            return logits
        else:
            # Inference mode
            generated = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

            hidden = None
            for _ in range(max_len):
                if generated.size(1) == 0:
                    decoder_input = torch.zeros(batch_size, 1, embed_dim, device=device)
                else:
                    prev_tokens = self.embedding(generated[:, -1:])

                # Combine with event context
                context = event_enc.unsqueeze(1)
                combined = torch.cat([prev_tokens, context], dim=-1)

                output, hidden = self.decoder(combined, hidden)
                logits = self.output_proj(output)
                next_token = logits.argmax(-1)
                generated = torch.cat([generated, next_token], dim=1)

            return generated
```

### Event Detection and Classification

```python
class SportsEventDetector:
    """
    Detect and classify significant game events for commentary.
    """
    def __init__(self, event_types):
        self.event_types = event_types

    def detect_from_tracking(self, tracking_frame, prev_frame):
        """
        Detect events from player tracking data.
        """
        events = []

        # Shot detection
        shot_event = self.detect_shot(tracking_frame, prev_frame)
        if shot_event:
            events.append(shot_event)

        # Pass detection
        pass_event = self.detect_pass(tracking_frame, prev_frame)
        if pass_event:
            events.append(pass_event)

        # Tackle detection
        tackle_event = self.detect_tackle(tracking_frame, prev_frame)
        if tackle_event:
            events.append(tackle_event)

        return events

    def detect_shot(self, current_frame, prev_frame):
        """
        Detect when a player takes a shot.
        """
        # Ball velocity spike indicates shot
        ball_vel = np.linalg.norm(current_frame.ball_velocity)

        if ball_vel > 25:  # m/s threshold for shot
            shooter = self.identify_ball_controller(current_frame)
            return {
                'type': 'shot',
                'player': shooter,
                'location': current_frame.ball_position,
                'velocity': ball_vel,
                'on_target': self.is_on_target(current_frame)
            }
        return None

    def detect_tackle(self, current_frame, prev_frame):
        """
        Detect successful tackles.
        """
        # Proximity of defender to ball carrier + ball transition
        for player in current_frame.defenders:
            dist_to_ball = np.linalg.norm(player.position - current_frame.ball_position)
            if dist_to_ball < 2:  # meters
                prev_dist = np.linalg.norm(
                    prev_frame.player_position(player.id) - prev_frame.ball_position
                )
                if prev_dist > 3 and dist_to_ball < 2:
                    return {
                        'type': 'tackle',
                        'player': player.id,
                        'location': player.position
                    }
        return None
```

### Template-Based with Neural Enhancement

Modern commentary systems often combine templates with neural generation:

```python
class HybridCommentarySystem:
    """
    Template-based commentary with neural enhancement.
    """
    def __init__(self):
        self.templates = self.load_templates()

        # Small language model for variety
        self.paraphraser = load_small_lm('gpt2')

    def generate_commentary(self, event, context):
        """
        Generate commentary for a detected event.
        """
        # Select base template
        template = self.select_template(event, context)

        # Fill slots
        filled = template.format(
            player=self.get_player_name(event['player']),
            location=self.describe_location(event['location']),
            team=self.get_team_name(event['team'])
        )

        # Enhance with paraphrasing for variety
        if np.random.random() < 0.3:  # 30% chance of neural enhancement
            enhanced = self.paraphraser.generate(filled, max_length=50)
            return enhanced
        return filled

    def select_template(self, event, context):
        """
        Select appropriate template based on event and game state.
        """
        templates = self.templates[event['type']]

        # Filter by game context
        if context['score_differential'] > 2:
            templates = [t for t in templates if 'desperation' not in t['tags']]
        elif context['time_remaining'] < 5 * 60:
            templates = [t for t in templates if 'late' in t['tags']]

        return np.random.choice(templates)
```

---

## Sentiment Analysis for Fan Reactions

### Social Media Sports Sentiment

Understanding fan reactions requires analyzing millions of social media posts:

```python
import torch.nn as nn

class SportsSentimentClassifier(nn.Module):
    """
    Fine-tuned sentiment classifier for sports text.
    """
    def __init__(self, base_model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.sentiment_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # negative, neutral, positive
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # joy, anger, sadness, surprise, etc.
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token

        sentiment = self.sentiment_head(pooled)
        emotions = self.emotion_head(pooled)

        return sentiment, emotions
```

### Real-Time Sentiment Aggregation

```python
class SentimentAggregator:
    """
    Aggregate sentiment across fanbase in real-time.
    """
    def __init__(self, team_id):
        self.team_id = team_id
        self.classifier = load_sentiment_model()
        self.stream = TwitterStream()

    def track_team_sentiment(self, time_window_minutes=5):
        """
        Track real-time sentiment for a team.
        """
        tweets = self.stream.filter_team(self.team_id, recent_minutes=time_window_minutes)

        sentiments = []
        for tweet in tweets:
            sentiment, emotions = self.classifier(tweet)
            sentiments.append({
                'sentiment': sentiment,  # -1, 0, 1
                'emotions': emotions,
                'engagement': tweet['likes'] + tweet['retweets'],
                'timestamp': tweet['created_at']
            })

        return self.aggregate(sentiments)

    def aggregate(self, sentiments):
        """
        Compute aggregate statistics.
        """
        if not sentiments:
            return {'sentiment_score': 0, 'dominant_emotion': 'neutral'}

        weighted_sentiment = sum(
            s['sentiment'] * np.log1p(s['engagement'])
            for s in sentiments
        ) / sum(np.log1p(s['engagement']) for s in sentiments)

        emotion_counts = {}
        for s in sentiments:
            dominant = s['emotions'].argmax()
            emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1

        return {
            'sentiment_score': weighted_sentiment,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get),
            'volume': len(sentiments),
            'intensity': np.std([s['sentiment'] for s in sentiments])
        }
```

---

## Automated Match Report Writing

### Generating Structured Summaries

```python
class MatchReportGenerator:
    """
    Generate written match reports from structured data.
    """
    def __init__(self, llm='gpt-4'):
        self.llm = llm

    def generate_report(self, match_data):
        """
        Generate full match report.

        match_data contains: score, events, stats, context
        """
        # Build structured input for LLM
        input_prompt = self.build_report_prompt(match_data)

        # Generate report sections
        sections = {
            'headline': self.generate_headline(match_data),
            'summary': self.generate_summary(match_data),
            'key_moments': self.generate_key_moments(match_data),
            'player_ratings': self.generate_player_ratings(match_data),
            'tactical_analysis': self.generate_tactical(match_data),
            'quotes': self.generate_quotes(match_data)
        }

        return sections

    def generate_headline(self, match_data):
        """
        Generate compelling match headline.
        """
        prompt = f"""
        Generate a headline for this match:
        {match_data['home_team']} {match_data['home_score']} - {match_data['away_score']} {match_data['away_team']}

        Key story: {match_data['key_story']}  # e.g., "upset victory", "record broken"

        Requirements:
        - Under 15 words
        - Captures the main narrative
        - Use active voice
        """
        return self.llm.generate(prompt, max_tokens=20)

    def generate_key_moments(self, match_data):
        """
        Write descriptions of key moments in the match.
        """
        moments = []
        for event in match_data['significant_events']:
            description = self.llm.generate(
                f"Describe this moment in 2-3 sentences: {event}",
                max_tokens=100
            )
            moments.append({
                'timestamp': event['time'],
                'description': description,
                'importance': event['importance']
            })
        return moments
```

### Data-to-Text Pipeline

```python
class DataToTextPipeline:
    """
    Convert structured match statistics into natural language.
    """
    def __init__(self):
        self.stat_descriptions = {
            'possession': "{team} dominated possession with {percent}% of the ball",
            'shots': "{team} attempted {count} shots, {on_target} on target",
            'passes': "{team} completed {count} passes with {accuracy}% accuracy",
            'fouls': "{team} committed {count} fouls"
        }

    def describe_stat(self, team_name, stat_type, value, context=None):
        """
        Generate natural language description of a statistic.
        """
        template = self.stat_descriptions.get(stat_type, "{team} had {value}")

        description = template.format(
            team=team_name,
            percent=value.get('percent', 0),
            count=value.get('count', 0),
            on_target=value.get('on_target', 0),
            accuracy=value.get('accuracy', 0)
        )

        # Add contextual flavor
        if context:
            if value.get('percent', 0) > 70:
                description = f"In complete control, {description.lower()}"
            elif value.get('percent', 0) < 30:
                description = f"Under pressure, {description.lower()}"

        return description
```

---

## Player Interview Analysis

### Extracting Insights from Press Conferences

```python
class PressConferenceAnalyzer:
    """
    Analyze player and coach press conferences.
    """
    def __init__(self):
        self.ner = load_ner_model()  # Named entity recognition
        self.summarizer = load_summarizer()

    def extract_key_themes(self, transcript):
        """
        Identify main themes from press conference transcript.
        """
        # Topic modeling with sports-specific dictionary
        sports_topics = {
            'tactical': ['formation', 'system', 'strategy', 'press', 'defend'],
            'performance': ['good', 'bad', 'improvement', 'effort', 'quality'],
            'injury': ['fitness', 'injury', 'recovery', 'match sharpness'],
            'future': ['next game', 'season', 'future', 'contract'],
            'team': ['teammates', 'team', 'chemistry', 'communication']
        }

        theme_scores = {topic: 0 for topic in sports_topics}
        for sentence in transcript:
            for topic, keywords in sports_topics.items():
                if any(kw in sentence.lower() for kw in keywords):
                    theme_scores[topic] += 1

        return {
            'primary_theme': max(theme_scores, key=theme_scores.get),
            'theme_distribution': theme_scores,
            'key_quotes': self.extract_notable_quotes(transcript)
        }

    def extract_notable_quotes(self, transcript, n=5):
        """
        Extract most notable quotes based on semantic intensity.
        """
        scored_quotes = []
        for i, line in enumerate(transcript):
            if len(line) > 30:  # Filter very short responses
                intensity = self.compute_intensity(line)
                scored_quotes.append({
                    'quote': line,
                    'speaker': transcript[i-1] if i > 0 else 'unknown',
                    'intensity': intensity
                })

        # Return top n by intensity
        return sorted(scored_quotes, key=lambda x: x['intensity'], reverse=True)[:n]
```

---

## Practical Applications

### Fantasy Sports Insights

```python
class FantasySportsNLP:
    """
    Generate fantasy sports insights using NLP.
    """
    def __init__(self):
        self.entity_linker = load_player_linker()

    def generate_matchup_analysis(self, player, opponent):
        """
        Generate fantasy-relevant analysis of a player vs opponent matchup.
        """
        # Find historical performance against similar defenses
        similar_defenses = self.find_similar_defenses(opponent)
        historical_performance = self.get_player_history_vs(player, similar_defenses)

        return {
            'projected_points': self.project_fantasy_points(player, opponent),
            'strengths_against_this_defense': self.identify_strengths(player, opponent),
            'risks': self.identify_risks(player, opponent),
            'ownership_recommendation': self.should_start_sit(player, opponent)
        }
```

---

## Summary

- Sports commentary generation combines event detection, template selection, and neural enhancement
- Sentiment analysis at scale enables real-time fan reaction monitoring
- Automated report writing transforms structured match data into engaging narratives
- Player interview analysis extracts key themes and notable quotes
- Sports NLP requires domain-specific vocabulary handling and entity recognition
- Hybrid systems combining rules and neural generation provide the best quality/diversity balance

---

## What's Next

Lesson 10 explores **AI for sports broadcasting and fan engagement** — how AI powers automated camera systems, real-time statistics overlays, interactive fan experiences, and the transformation of how fans consume sports content.