synthetic_sentences = [
    # From sample1 (positive)
    ("The staff was insanely friendly and knowledgeable", 2, 1),
    ("Matt was an awesome employee", 2, 1),
    ("The climbs were creative and unique", 2, 2),
    ("I drove an hour and a half and it was totally worth it", 2, 4),
    ("Loved the holiday climbs", 2, 2),
    ("Everything felt very creative", 2, 2),
    ("Can’t wait to go back again!", 2, 4),
    ("This place had a great vibe", 2, 4),
    ("Staff made me feel welcome and informed", 2, 1),
    ("The route setting was fun and interesting", 2, 2),

    # From sample2 (positive, neutral)
    ("Tried it for the day and had fun", 2, 4),
    ("Felt welcomed as a beginner", 2, 1),
    ("The pricing was reasonable for a full day", 2, 3),
    ("They offer routes for all skill levels", 2, 2),
    ("Super fun place to climb", 2, 4),
    ("Good spot for new climbers", 2, 2),
    ("Staff was welcoming right when I arrived", 2, 1),
    ("Not too expensive for a day pass", 2, 3),
    ("They made me feel like I belonged", 2, 1),
    ("Beginner-friendly routes were great", 2, 2),

    # From sample3 (negative)
    ("This gym is overly profit-hungry", 0, 3),
    ("They make canceling your membership difficult", 0, 3),
    ("Prices were much higher than expected", 0, 3),
    ("Worst pricing I've seen in Central Texas", 0, 3),
    ("Staff refuses to communicate about cancellations", 0, 1),
    ("They redirect you constantly when canceling", 0, 1),
    ("User-unfriendly cancellation process", 0, 3),
    ("Customer service was really frustrating", 0, 1),
    ("Feels like a money grab", 0, 3),
    ("Higher prices than Houston or Killeen gyms", 0, 3),

    # Mixed & Additional Synthetic Variants
    ("The bathrooms and lounge area were spotless", 2, 0),
    ("Great facility with nice amenities", 2, 0),
    ("The gym was too crowded", 0, 0),
    ("I loved the bouldering problems", 2, 2),
    ("Climbing wall variety was excellent", 2, 2),
    ("Front desk staff could’ve been nicer", 1, 1),
    ("Didn’t like the lighting in the facility", 0, 0),
    ("Membership options are confusing", 1, 3),
    ("Very clean and spacious gym", 2, 0),
    ("The climbing holds were dirty", 0, 0),
    ("Fun routes but the music was too loud", 1, 2),
    ("Helpful staff explained everything clearly", 2, 1),
    ("No AC made it uncomfortable", 0, 0),
    ("Felt safe and supported as a new climber", 2, 1),
    ("The pricing page was misleading", 0, 3),
    ("Loved the auto-belays", 2, 2),
    ("So many route styles to try!", 2, 2),
    ("Friendly community and atmosphere", 2, 4),
    ("Wish they had more lockers", 1, 0)
]

SAMPLE_DATA = {}
SAMPLE_DATA['sentences'] = [s[0] for s in synthetic_sentences]
SAMPLE_DATA['labels_classification'] = [s[2] for s in synthetic_sentences]
SAMPLE_DATA['labels_sentiment'] = [s[1] for s in synthetic_sentences]