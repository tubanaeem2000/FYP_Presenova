"""
Main Flask Application Hub
Role: Initialize Flask app, configure database and authentication, register blueprints, and expose health-check endpoint.

Key Features:
- SQLAlchemy ORM for database management
- JWT-Extended for secure token-based authentication
- CORS enabled for frontend integration
- Blueprint-based modular architecture
- Comprehensive error handling
"""
import eventlet
eventlet.monkey_patch()
import sys

# Ensure site-packages from global user site do not leak into sys.path on Windows
sys.path = [p for p in sys.path if "AppData\\Roaming\\Python" not in p and "AppData/Roaming/Python" not in p]

# Force stdout/stderr to use UTF-8 encoding to prevent UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
from datetime import timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global SocketIO instance
socketio = SocketIO()

# Import blueprints
from auth import auth_bp, register_jwt_error_handlers, signup, login, firebase_login, get_current_user
from phase_two import phase_two_bp
from phase_four import phase_four_bp
from phase_five import phase_five_bp
from phase_live import phase_live_bp, init_socketio_events
from routes.presentation_rewriter import presentation_rewriter_bp
from routes.question_generator import question_generator_bp
from routes.presentation_generator import presentation_generator_bp
from services.download_service import MAX_UPLOAD_BYTES
import logging
import time

logger = logging.getLogger(__name__)

def prewarm_ml_models():
    """
    Pre-warm ML models at startup in background thread to eliminate per-request model loading latency.
    Loads spaCy, SentenceTransformer, sklearn PresentationScorer, and Coach Intent Classifier into RAM.
    """
    start = time.time()
    logger.info("[PERF] Pre-warming ML models in background...")

    try:
        from nlp_module.scoring_model import load_scoring_models
        load_scoring_models()
    except Exception as e:
        logger.error(f"[PERF] Could not pre-warm scoring model: {e}", exc_info=True)

    try:
        from services.viva_rag_engine import _load_sentence_model, _load_spacy
        _load_sentence_model()
        _load_spacy()
    except Exception as e:
        logger.error(f"[PERF] Could not pre-warm SentenceTransformer/spaCy: {e}", exc_info=True)

    try:
        from services.coach_intent_engine import _get_intent_classifier
        _get_intent_classifier()
    except Exception as e:
        logger.error(f"[PERF] Could not pre-warm intent classifier: {e}", exc_info=True)

    elapsed = time.time() - start
    logger.info(f"[PERF] All ML models pre-warmed successfully in {elapsed:.3f}s")


def create_app():
    """
    Factory function to create and configure the Flask application.
    
    Initializes:
    - JWT authentication
    - CORS
    - Error handlers
    - Blueprints
    - MongoDB connection check
    - Async pre-warmed ML Models
    """
    import threading
    
    # ===== CREATE FLASK APP =====
    app = Flask(__name__)

    # Pre-warm ML models in background thread so server starts instantly
    threading.Thread(target=prewarm_ml_models, daemon=True).start()

    # ===== JWT CONFIGURATION =====
    # CRITICAL: In production, use a strong secret key from environment variables
    jwt_secret_key = os.getenv('JWT_SECRET_KEY', '').strip()
    if not jwt_secret_key:
        if os.getenv('FLASK_ENV', 'development').lower() == 'production':
            raise RuntimeError('JWT_SECRET_KEY must be configured in production.')
        jwt_secret_key = 'development-only-change-me'
    
    app.config['JWT_SECRET_KEY'] = jwt_secret_key
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
    
    # Initialize JWT with the app
    jwt = JWTManager(app)

    # ===== CORS CONFIGURATION =====
    DEFAULT_ALLOWED_ORIGINS = ['http://localhost:3000', 'http://localhost:5173']
    cors_origins_env = os.getenv('CORS_ORIGINS', '').strip()
    if cors_origins_env and cors_origins_env != '*':
        parsed_origins = [o.strip() for o in cors_origins_env.split(',') if o.strip()]
    else:
        parsed_origins = DEFAULT_ALLOWED_ORIGINS

    CORS(
        app,
        resources={r"/*": {
            "origins": parsed_origins,
            "allow_headers": ["Content-Type", "Authorization"],
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        }},
    )

    # Flask enforces this before request handlers read multipart bodies. This
    # prevents oversized uploads from being copied to disk first.
    app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({
            'success': False,
            'message': f'File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.',
        }), 413

    # ===== SOCKET.IO CONFIGURATION =====
    socketio.init_app(app, cors_allowed_origins=parsed_origins)

    # ===== MONGO CONFIGURATION & VERIFICATION =====
    with app.app_context():
        try:
            from models import db
            if db is not None:
                print("[INIT OK] Database layer initialized successfully")
        except Exception as e:
            print(f"[INIT FAIL] Database initialization failed: {str(e)}")

    # ===== REGISTER JWT ERROR HANDLERS =====
    # Handles expired, invalid, and missing JWT tokens
    register_jwt_error_handlers(app)

    # ===== REGISTER BLUEPRINTS =====
    # Phase 1: Authentication
    app.register_blueprint(auth_bp)
    
    # Compatibility blueprint for /auth without /api prefix
    from flask import Blueprint as BP
    auth_compat_bp = BP('auth_compat', __name__, url_prefix='/auth')
    auth_compat_bp.add_url_rule('/firebase-login', 'firebase_login_compat', firebase_login, methods=['POST', 'OPTIONS'])
    auth_compat_bp.add_url_rule('/login', 'login_compat', login, methods=['POST', 'OPTIONS'])
    auth_compat_bp.add_url_rule('/signup', 'signup_compat', signup, methods=['POST', 'OPTIONS'])
    auth_compat_bp.add_url_rule('/me', 'me_compat', get_current_user, methods=['GET', 'OPTIONS'])
    app.register_blueprint(auth_compat_bp)
    
    # Phase 2: Document Analysis
    app.register_blueprint(phase_two_bp)
    
    # Phase 4: Speech Analysis
    app.register_blueprint(phase_four_bp)
    
    # Phase 5: AI Coach / Practice Mode
    app.register_blueprint(phase_five_bp)

    # Phase Live: Presentation Coach & Live Analyzer
    app.register_blueprint(phase_live_bp)
    init_socketio_events(socketio)

    # New Feature: AI Presentation Rewriter
    app.register_blueprint(presentation_rewriter_bp)

    # New Feature: Viva Question Generator
    app.register_blueprint(question_generator_bp)

    # New Feature: AI Presentation Generator
    app.register_blueprint(presentation_generator_bp)

    # ===== HEALTH-CHECK ENDPOINT =====
    @app.route('/', methods=['GET'])
    def health_check():
        """Health check endpoint to verify the service is running."""
        return jsonify({
            "status": "running",
            "service": "Presenova AI Presentation Platform",
            "version": "1.1.0",
            "database": "Firebase Firestore"
        }), 200

    @app.route('/api/health', methods=['GET'])
    def api_health():
        """Render.com health check endpoint."""
        return jsonify({"status": "ok"}), 200

    return app


if __name__ == '__main__':
    app = create_app()

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))

    # Run the Flask development server wrapped with Socket.IO
    debug = os.getenv('FLASK_DEBUG', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False, allow_unsafe_werkzeug=True)

