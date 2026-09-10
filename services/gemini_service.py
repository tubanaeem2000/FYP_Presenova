"""
Gemini Service — Compatibility Shim

This file is maintained for backward compatibility.
All new code should import from services.ai instead.

Legacy symbols (functions, constants) are kept here so that existing
imports across the codebase continue to work without changes.
"""

import json
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── SDK Imports (graceful degradation) ──────────────────────────────────────
try:
    import google.genai as current_genai
except ImportError:
    current_genai = None

try:
    import google.generativeai as legacy_genai
except ImportError:
    legacy_genai = None


# ── Environment helpers ─────────────────────────────────────────────────────
def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning('[gemini_service] Invalid integer for %s; using %s.', name, default)
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning('[gemini_service] Invalid number for %s; using %s.', name, default)
        return default


# ── Configuration ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash').strip()
MODEL_CANDIDATES = [
    model for model in [
        GEMINI_MODEL,
        *os.getenv('GEMINI_FALLBACK_MODELS', 'gemini-3.5-flash-lite').split(','),
    ] if model.strip()
]
GEMINI_TIMEOUT_SECONDS = max(1.0, _env_float('GEMINI_TIMEOUT_SECONDS', 45.0))
GEMINI_MAX_RETRIES = max(1, _env_int('GEMINI_MAX_RETRIES', 2))
MAX_REWRITE_PROMPT_CHARS = max(1000, _env_int('GEMINI_MAX_REWRITE_PROMPT_CHARS', 100000))
MAX_ANALYSIS_PROMPT_CHARS = max(1000, _env_int('GEMINI_MAX_ANALYSIS_PROMPT_CHARS', 120000))
GEMINI_OFFLINE = _env_bool('PRESENTATION_REWRITER_OFFLINE', False)

_new_client = None
if GEMINI_API_KEY and not GEMINI_OFFLINE and current_genai is not None:
    try:
        _new_client = current_genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:
        logger.warning('[gemini_service] Current Gemini SDK configuration failed: %s', exc)

if GEMINI_API_KEY and not GEMINI_OFFLINE and _new_client is None and legacy_genai is not None:
    try:
        legacy_genai.configure(api_key=GEMINI_API_KEY, transport='rest')
    except Exception as exc:
        logger.warning('[gemini_service] Legacy Gemini SDK configuration failed: %s', exc)

gemini_available = bool(
    GEMINI_API_KEY and not GEMINI_OFFLINE and (_new_client is not None or legacy_genai is not None)
)
if not gemini_available:
    logger.info('[gemini_service] Gemini disabled/unavailable; deterministic local fallback is active.')


# ── JSON cleaning ───────────────────────────────────────────────────────────
def _clean_json_response(raw: str) -> str:
    """Remove markdown and return the first valid JSON value in a response."""
    if not isinstance(raw, str):
        return ''
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in '{[':
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
            return text[index:index + end].strip()
        except json.JSONDecodeError:
            continue
    return text


# ── Local fallback text polish ──────────────────────────────────────────────
def _polish_text_fallback(text: str) -> str:
    """Make conservative local corrections when Gemini is unavailable."""
    if not text or not text.strip():
        return text
    polished = text.strip()
    if polished[0].islower():
        polished = polished[0].upper() + polished[1:]
    contractions = {
        r"\bdon'?t\b": 'do not', r"\bcan'?t\b": 'cannot', r"\bwon'?t\b": 'will not',
        r"\bisn'?t\b": 'is not', r"\baren'?t\b": 'are not', r"\bwasn'?t\b": 'was not',
        r"\bweren'?t\b": 'were not', r"\bhaven'?t\b": 'have not', r"\bhasn'?t\b": 'has not',
        r"\bhadn'?t\b": 'had not', r"\bdoesn'?t\b": 'does not', r"\bdidn'?t\b": 'did not',
        r"\bwouldn'?t\b": 'would not', r"\bshouldn'?t\b": 'should not',
        r"\bcouldn'?t\b": 'could not', r"\bit'?s\b": 'it is', r"\bwe'?re\b": 'we are',
        r"\bthey'?re\b": 'they are', r"\byou'?re\b": 'you are', r"\bthere'?s\b": 'there is',
        r"\bthat'?s\b": 'that is', r"\bwhat'?s\b": 'what is', r"\blet'?s\b": 'let us',
        r"\bi'?m\b": 'I am', r"\bwe'?ll\b": 'we will', r"\bthey'?ll\b": 'they will',
        r"\byou'?ll\b": 'you will', r"\bi'?ve\b": 'I have', r"\bwe'?ve\b": 'we have',
        r"\bthey'?ve\b": 'they have', r"\byou'?ve\b": 'you have',
    }
    for pattern, replacement in contractions.items():
        polished = re.sub(pattern, replacement, polished, flags=re.IGNORECASE)
    weak_phrases = [
        r'\bas you can see\b', r'\bit is important to note that\b',
        r'\bits worth mentioning that\b', r'\bin this (section|slide|presentation)\b',
        r'\bi would like to\b', r'\blet me walk you through\b',
        r'\bas previously mentioned\b', r'\bas discussed earlier\b',
        r'\bi will now discuss\b', r'\bin order to\b',
        r'\bthe following\b', r'\bthe above (mentioned )?(slide|section|chart|table)\b',
    ]
    for phrase in weak_phrases:
        polished = re.sub(phrase, '', polished, flags=re.IGNORECASE).strip()
    polished = re.sub(r'[ \t]+', ' ', polished)
    polished = re.sub(r'\s+([.,!?:;])', r'\1', polished)
    if polished and not polished.endswith(('.', '!', '?', ':', ';', ',')):
        if len(polished) > 4:
            polished += '.'
    return polished


def _paragraph_text(paragraph: dict) -> str:
    return str(paragraph.get('text', ''))


def _infer_slide_type(slide: dict) -> str:
    """Heuristic to classify the role/purpose of a slide based on its title and content."""
    title = (slide.get('title') or '').lower().strip()
    full_text = ' '.join(
        _paragraph_text(p) for tb in slide.get('textboxes', [])
        for p in tb.get('paragraphs', [])
    ).lower()
    type_signals = [
        (r'\b(agenda|outline|overview|roadmap|table of contents|what we will cover)\b', 'Agenda / Outline'),
        (r'\b(introduction|intro|background|context|overview|setting the stage)\b', 'Introduction / Background'),
        (r'\b(problem|challenge|pain point|issue|gap|difficulty|obstacle)\b', 'Problem Statement'),
        (r'\b(objective|goal|aim|purpose|mission|vision|target)\b', 'Objectives / Goals'),
        (r'\b(literature|related work|prior work|previous research|state of the art|background study)\b', 'Literature Review'),
        (r'\b(methodology|approach|method|technique|framework|strategy|process|procedure)\b', 'Methodology / Approach'),
        (r'\b(architecture|system design|system overview|system architecture|infrastructure|platform)\b', 'Architecture / System Design'),
        (r'\b(workflow|pipeline|flow|process flow|steps|stages|phase)\b', 'Workflow / Process'),
        (r'\b(comparison|compare|vs\.|versus|alternative|benchmark|evaluation)\b', 'Comparison / Evaluation'),
        (r'\b(result|finding|outcome|discovery|observation|insight)\b', 'Results / Findings'),
        (r'\b(conclusion|summary|takeaway|key point|wrap.up|recap|closing)\b', 'Conclusion / Summary'),
        (r'\b(future work|next steps|future direction|upcoming|planned)\b', 'Future Work / Next Steps'),
        (r'\b(thank|questions\?|qa|q&a|discuss|contact|get in touch)\b', 'Thank You / Q&A'),
        (r'\b(timeline|schedule|milestone|deadline|roadmap|gantt)\b', 'Timeline / Milestones'),
        (r'\b(feature|capability|functionality|benefit|advantage|key offering)\b', 'Features / Benefits'),
        (r'\b(statistic|data|metric|kpi|number|figure|percentage|chart)\b', 'Data / Statistics'),
        (r'\b(recommendation|proposal|suggestion|action plan|call to action)\b', 'Recommendations / Call to Action'),
        (r'\b(case study|example|scenario|use case|demonstration|demo)\b', 'Case Study / Example'),
        (r'\b(team|about us|our team|who we are|organization|company)\b', 'About / Team'),
    ]
    for pattern, label in type_signals:
        if re.search(pattern, title) or re.search(pattern, full_text[:200]):
            return label
    textbox_count = len(slide.get('textboxes', []))
    total_text_length = sum(
        len(_paragraph_text(p)) for tb in slide.get('textboxes', [])
        for p in tb.get('paragraphs', [])
    )
    if textbox_count <= 1 and total_text_length < 60:
        return 'Section Divider / Title Slide'
    if textbox_count >= 5 or total_text_length > 800:
        return 'Detailed Content Section'
    return 'Content Section'


def _heuristic_presentation_context(slides: list[dict]) -> dict:
    """Local fallback when Gemini is unavailable for the analysis phase."""
    titles = [slide.get('title', '') for slide in slides]
    all_text = ' '.join(
        _paragraph_text(p) for slide in slides
        for tb in slide.get('textboxes', [])
        for p in tb.get('paragraphs', [])
    )
    topic = titles[0] if titles and titles[0].strip() else 'Untitled Presentation'
    roles = {str(slide.get('slide_number')): _infer_slide_type(slide) for slide in slides}
    word_count = len(all_text.split())
    technical_level = 'academic' if word_count > 500 else 'intermediate'
    return {
        'overall_topic': topic,
        'main_objective': 'Inform and persuade the audience.',
        'presentation_type': 'Conference / academic presentation',
        'technical_level': technical_level,
        'audience': 'Subject-matter audience',
        'story_flow': 'Introduction → body → conclusion.',
        'key_themes': [],
        'tone_guidance': 'Professional, clear, and concise.',
        'slide_roles': roles,
    }


def _fallback_local_rewrite(slides: list[dict], grammar_issues_summary: str = '') -> list[dict]:
    """Rewrite supported targets locally while preserving slide structure."""
    del grammar_issues_summary
    result = []
    for slide in slides:
        textboxes = []
        for textbox in slide.get('textboxes', []):
            textboxes.append({
                'shape_index': textbox.get('shape_index'),
                'paragraphs': [
                    _polish_text_fallback(p.get('text', '') if isinstance(p, dict) else str(p))
                    for p in textbox.get('paragraphs', [])
                ],
            })
        tables = []
        for table in slide.get('tables_data', []):
            cells = []
            for cell in table.get('cells', []):
                cells.append({
                    'row_index': cell.get('row_index'),
                    'column_index': cell.get('column_index'),
                    'paragraphs': [
                        _polish_text_fallback(p.get('text', '') if isinstance(p, dict) else str(p))
                        for p in cell.get('paragraphs', [])
                    ],
                })
            tables.append({'shape_index': table.get('shape_index'), 'cells': cells})
        charts = []
        for chart in slide.get('charts_data', []):
            charts.append({
                'shape_index': chart.get('shape_index'),
                'title_paragraphs': [
                    _polish_text_fallback(p.get('text', '') if isinstance(p, dict) else str(p))
                    for p in chart.get('title_paragraphs', [])
                ],
            })
        rewritten = {
            'slide_number': slide.get('slide_number'),
            'textboxes': textboxes,
            'charts': charts,
        }
        if 'tables_data' in slide:
            rewritten['tables'] = tables
        result.append(rewritten)
    return result


def build_presentation_context_prompt(slides: list[dict]) -> str:
    """Build a prompt to analyse the COMPLETE presentation before rewriting."""
    from services.ai.prompts import build_presentation_context_prompt as _new
    return _new(slides)


def call_gemini_presentation_context(slides: list[dict]) -> dict:
    """Return a validated presentation-context summary."""
    if not slides or not gemini_available:
        return _heuristic_presentation_context(slides)
    try:
        from services.analysis.holistic import HolisticAnalyzer
        from services.ai import get_provider
        provider = get_provider()
        analyzer = HolisticAnalyzer(provider=provider)
        return analyzer.analyze(slides)
    except Exception:
        return _heuristic_presentation_context(slides)


def build_rewrite_prompt(
    slides: list[dict],
    grammar_issues_summary: str = '',
    presentation_context: Optional[dict] = None,
) -> str:
    """Build the Gemini rewrite prompt (delegates to new prompts module)."""
    from services.ai.prompts import build_rewrite_prompt as _new
    return _new(slides, grammar_issues_summary, presentation_context)


def call_gemini_quality_analysis(text: str, filename: str = '') -> dict:
    """Legacy compatibility: analyze text quality using Gemini."""
    from services.analysis.per_slide import PerSlideAnalyzer
    try:
        from services.ai import get_provider
        provider = get_provider()
        analyzer = PerSlideAnalyzer(provider=provider)
    except Exception:
        analyzer = PerSlideAnalyzer()
    return analyzer.analyze_presentation(text, filename)


def call_gemini_rewrite(
    slides: list[dict],
    grammar_issues_summary: str = '',
    presentation_context: Optional[dict] = None,
) -> list[dict]:
    """Legacy compatibility: rewrite slides using Gemini."""
    from services.rewrite.executor import RewriteExecutor
    executor = RewriteExecutor()
    rewritten, _ = executor.execute_rewrite(slides, grammar_issues_summary, presentation_context)
    return rewritten


def validate_rewritten_slides(
    original_slides: list[dict],
    rewritten_slides: list[dict],
) -> None:
    """Legacy compatibility: validate rewritten slide structure."""
    from services.validation.structural import StructuralValidator
    validator = StructuralValidator()
    result = validator.validate_slides(original_slides, rewritten_slides)
    if not result.get('valid', True):
        issues = result.get('issues', [])
        if issues:
            raise ValueError(f'Structural validation failed: {issues[0].get("message", "unknown")}')


def estimate_rewrite_token_budget(slides: list[dict]) -> dict:
    """Estimate token requirements for rewriting slides."""
    from services.rewrite.planner import RewritePlanner
    planner = RewritePlanner()
    return planner.estimate_token_budget(slides)


def get_model_usage_summary() -> dict:
    """Get usage summary."""
    return {'provider': 'gemini', 'available': gemini_available}


__all__ = [
    'call_gemini_quality_analysis',
    'call_gemini_rewrite',
    'validate_rewritten_slides',
    'build_rewrite_prompt',
    'build_presentation_context_prompt',
    'call_gemini_presentation_context',
    '_clean_json_response',
    '_polish_text_fallback',
    '_fallback_local_rewrite',
    '_heuristic_presentation_context',
    '_infer_slide_type',
    'gemini_available',
    'GEMINI_API_KEY',
    'MODEL_CANDIDATES',
    'MAX_REWRITE_PROMPT_CHARS',
    'MAX_ANALYSIS_PROMPT_CHARS',
    'estimate_rewrite_token_budget',
    'get_model_usage_summary',
]

