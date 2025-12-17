import feedparser
import requests
import json
import os
import pyttsx3


def fetch_techcrunch():
    feed_url = "https://techcrunch.com/feed/"
    feed = feedparser.parse(feed_url)

    articles = []

    for entry in feed.entries[:5]:
        article = {
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        }
        articles.append(article)

    return articles


def fetch_hackernews():
    top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(top_stories_url)
    story_ids = response.json()[:5]

    articles = []

    for story_id in story_ids:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        item = requests.get(item_url).json()

        article = {
            "title": item.get("title"),
            "link": item.get("url"),
            "published": item.get("time")
        }
        articles.append(article)

    return articles

def save_news_to_file(news, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(news, f, indent=2)


def load_previous_news(filename):
    if not os.path.exists(filename):
        return []
    
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_new_news(today_news, previous_news):
    previous_titles = set(item["title"] for item in previous_news)

    new_items = []
    for item in today_news:
        if item["title"] not in previous_titles:
            new_items.append(item)

    return new_items

def score_news(news_items):
    IMPORTANT_KEYWORDS = [
        "release", "launch", "announces", "regulation", "ban",
        "security", "breach", "lawsuit", "earnings", "funding",
        "acquires", "acquisition", "policy", "breakthrough"
    ]

    scored_items = []

    for item in news_items:
        title = item["title"].lower()
        score = 0

        for keyword in IMPORTANT_KEYWORDS:
            if keyword in title:
                score += 1

        scored_items.append((score, item))

    # Sort by score (highest first)
    scored_items.sort(key=lambda x: x[0], reverse=True)

    return scored_items


def generate_daily_brief(top_news):
    if not top_news:
        return "Good morning.\n\nThere are no major AI or industry updates today."

    lines = []
    lines.append("Good morning.\n")
    lines.append("Here’s what actually matters today:\n")

    for idx, item in enumerate(top_news, start=1):
        enriched = enrich_title(item["title"])
        lines.append(f"{idx}. {enriched}")

    lines.append(
        "\nOverall, these updates show how AI continues to shape products, hiring, and company strategy."
    )

    return "\n".join(lines)


def enrich_title(title):
    title_lower = title.lower()

    companies = [
        "nvidia", "tesla", "google", "microsoft",
        "amazon", "meta", "openai", "apple"
    ]

    if any(company in title_lower for company in companies):
        return f"{title} — This signals increased AI adoption by major companies."

    if "interview" in title_lower or "hiring" in title_lower:
        return f"{title} — This may impact hiring and interview processes."

    if "ai" in title_lower or "artificial intelligence" in title_lower:
        return f"{title} — This reflects ongoing advances in AI usage."

    return f"{title} — General industry update."

def save_brief_to_file(brief, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(brief)


def generate_audio_from_text(text, output_file):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_file)
    engine.runAndWait()


if __name__ == "__main__":
    techcrunch_news = fetch_techcrunch()
    hackernews_news = fetch_hackernews()

    all_news = techcrunch_news + hackernews_news
    previous_news = load_previous_news("data/previous_news.json")

    new_news = detect_new_news(all_news, previous_news)

    # If no new news, fall back to current top news
    if new_news:
        candidate_news = new_news
        print("Using NEW updates for today's brief.\n")
    else:
        candidate_news = all_news
        print("No new updates detected — using top current news.\n")

    scored_news = score_news(candidate_news)
    top_news = [item for score, item in scored_news][:5]

    brief = generate_daily_brief(top_news)

    print("\nDAILY BRIEF:\n")
    print(brief)
    save_brief_to_file(brief, "output/daily_brief.txt")
    generate_audio_from_text(brief, "output/daily_brief.wav")



    # Update memory
    save_news_to_file(all_news, "data/previous_news.json")

