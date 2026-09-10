"""
Shared Configuration Constants for the AI Presentation Enhancement Platform.

Centralizes all numeric constants, thresholds, and configuration values
to eliminate duplication across modules.
"""

# ── Reading/Speaking Speed ──────────────────────────────────────────────────
# Average reading speed for presentations (words per minute)
READING_WORDS_PER_MINUTE = 220
# Average speaking speed for presentations (words per minute)
SPEAKING_WORDS_PER_MINUTE = 150
# Slow speaking speed (for estimates)
SLOW_SPEAKING_WPM = 130
# Fast speaking speed (for estimates)
FAST_SPEAKING_WPM = 170
# Additional transition time between slides (seconds)
SECONDS_PER_SLIDE_TRANSITION = 10
# Average time per slide for presenter explanation (seconds)
SECONDS_PER_SLIDE = 45

# ── Token Budget Management ─────────────────────────────────────────────────
SAFETY_MARGIN = 0.85          # Reserve 15% for output tokens
MAX_CONTEXT_TOKENS = 128000    # Maximum context window size
MAX_OUTPUT_TOKENS = 8192       # Maximum output tokens per request
CHARS_PER_TOKEN = 4            # Rough estimate: ~4 characters per token

# ── Per-mode Token Multipliers ──────────────────────────────────────────────
MODE_TOKEN_MULTIPLIERS = {
    "quick": 0.5,
    "professional": 1.0,
    "academic": 1.3,
}

# ── Per-mode Batch Sizes ────────────────────────────────────────────────────
MODE_BATCH_SIZES = {
    "quick": 20,
    "professional": 15,
    "academic": 10,
}

# ── Smart Filter Thresholds ─────────────────────────────────────────────────
SMART_FILTER_QUALITY_THRESHOLD = 0.85      # Skip text with quality >= 85%
SMART_FILTER_MIN_WORDS = 3                 # Skip text shorter than 3 words
SMART_FILTER_LABEL_MAX_LENGTH = 20         # Characters: treat as label if shorter
SMART_FILTER_TITLE_QUALITY_THRESHOLD = 0.80

# ── Duplicate Detection ────────────────────────────────────────────────────
MIN_PHRASE_LENGTH = 15                     # Min length for duplicate detection
MIN_DUPLICATE_OCCURRENCES = 2              # Min occurrences to flag as duplicate
SIMILARITY_THRESHOLD = 0.85                # Similarity threshold for fuzzy matching

# ── Grade Boundaries ────────────────────────────────────────────────────────
GRADE_BOUNDARIES = [
    (95, 'A+'), (90, 'A'), (85, 'A-'),
    (80, 'B+'), (75, 'B'), (70, 'B-'),
    (65, 'C+'), (60, 'C'), (55, 'C-'),
    (50, 'D+'), (45, 'D'), (40, 'D-'),
]

# ── Quality Levels ──────────────────────────────────────────────────────────
QUALITY_LEVELS = [
    (95, 'Excellent', 'Professional-grade presentation'),
    (85, 'Very Good', 'Near-professional presentation quality'),
    (75, 'Good', 'Solid presentation with minor improvements needed'),
    (65, 'Fair', 'Adequate but significant improvements possible'),
    (0, 'Needs Work', 'Substantial improvements recommended'),
]

# ── Flesch Reading Ease Thresholds ──────────────────────────────────────────
FLESCH_SCORES = {
    'very_easy': (90, 100),
    'easy': (80, 89),
    'fairly_easy': (70, 79),
    'standard': (60, 69),
    'fairly_difficult': (50, 59),
    'difficult': (30, 49),
    'very_confusing': (0, 29),
}

# ── Valid Modes and Tones ───────────────────────────────────────────────────
VALID_MODES = {'quick', 'professional', 'academic'}
VALID_TONES = {
    'professional', 'academic', 'business', 'technical',
    'executive', 'marketing', 'formal', 'simple_english',
}

# ── Priority Weights ─────────────────────────────────────────────────────────
PRIORITY_SCORES = {'high': 90, 'medium': 65, 'low': 40}
IMPACT_LEVELS = {
    'high': 'Significant improvement expected',
    'medium': 'Moderate improvement expected',
    'low': 'Minor improvement expected',
}

# ── Severity Thresholds ─────────────────────────────────────────────────────
SEVERITY_HIGH_CHANGE_RATIO = 0.5    # >50% text changed = major rewrite
SEVERITY_MODERATE_CHANGE_RATIO = 0.2  # >20% = moderate

# ── Fallback Defaults ───────────────────────────────────────────────────────
FALLBACK_MAX_RETRIES = 2
FALLBACK_FAILURE_THRESHOLD = 0.5  # Abort if >50% of items fail

# ── Gemini Provider Defaults ────────────────────────────────────────────────
GEMINI_DEFAULT_TIMEOUT = 45.0
GEMINI_DEFAULT_MAX_RETRIES = 2
GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"
GEMINI_DEFAULT_FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-pro"]

