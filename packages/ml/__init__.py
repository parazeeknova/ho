"""ho-ml — learning-to-rank, bandits, and outcome feedback for ho.

This package owns the closed-loop decision system: event stream, feature
substrate, ranking models, calibration, contextual bandits, and the Gmail
push ingestion that turns delayed recruiter mail into training signal.

Phases:
  0 — event infrastructure + Gmail Pub/Sub + attribution + snapshots
  1 — feature substrate (graph BEFORE LTR)
  2 — LTR shadow: separate LambdaMART ranker + binary classifiers
  3 — offline evaluation
  4 — live LTR (EV ranking, hard-constraint boundary)
  5 — contextual bandits (source / query / recommendation / board-routing)
  6 — structured exploration + counterfactual evaluation
  7 — autonomous optimization
"""

__version__ = "0.1.0"

# Version constants stamped on every decision event. Bump on each
# training/policy change so offline evaluation can reconstruct "what model
# made this decision".
RANKER_VERSION = "ranker_v1"
LGB_RANKER_VERSION = "lgb_ranker_v1"
CLASSIFIER_VERSION = "clf_v1"
FEATURE_VERSION = "features_v1"
POLICY_VERSION = "policy_v1"
BANDIT_VERSION = "bandit_v1"
GMAIL_CLASSIFIER_VERSION = "gmail_v1"
PROMPT_VERSION = "prompt_v1"
EMBEDDING_VERSION = "qwen_v1"

# Canonical event types. Every event is append-only; rewards arrive later.
# The distinction between "never seen / seen not selected / selected ignored /
# selected applied" is first-class — this is what fixes selection bias.
EVENT_TYPES = {
    # discovery & parsing
    "job_discovered",
    "job_parsed",
    "job_gated",
    # exposure (solves selection bias — #3, #4, #5)
    "job_exposed",  # job was shown in a ranking (impression member)
    "recommendation_exposed",  # an entire recommendation batch was shown
    "impression",  # one ranking decision (group of jobs)
    # ranking
    "job_ranked",
    # user actions (pairwise signal — #9)
    "job_clicked",
    "job_saved",
    "job_rejected_by_user",
    "job_accepted_by_user",
    "job_applied",
    # application lifecycle
    "application_submitted",
    "application_failed",
    "application_confirmed",  # email confirmation received
    # recruiter / email outcomes (delayed reward — Gmail push)
    "confirmation_email",
    "rejection_email",
    "screening_email",
    "interview",
    "technical_assessment",
    "onsite",
    "offer",
    "withdrawn",
    "recruiter_email",  # generic recruiter contact
    # legacy / compat
    "application_outcome",  # maps to application_outcomes table
}

# Reward hierarchy — staged, NOT binary. Each outcome also carries its
# semantic event; the number is for policy optimization only. (#10)
# Rewards are discounted by gamma**delta_days at training time; raw events
# are preserved.
REWARD_MAP: dict[str, float] = {
    "job_discovered": 0.01,
    "job_gated": 0.02,
    "job_exposed": 0.05,
    "recommendation_exposed": 0.05,
    "job_ranked": 0.0,  # not a reward event, just a decision
    "job_clicked": 0.10,
    "job_saved": 0.25,
    "job_rejected_by_user": -0.8,
    "job_applied": 0.50,
    "application_submitted": 0.50,
    "application_confirmed": 0.75,
    "application_failed": -0.25,
    "confirmation_email": 0.75,
    "recruiter_email": 2.0,
    "rejection_email": -1.0,
    "bad_job": -0.5,
    "screening_email": 5.0,
    "screening": 5.0,
    "technical_assessment": 8.0,
    "interview": 15.0,
    "onsite": 30.0,
    "offer": 100.0,
    "withdrawn": -0.5,
}

# Hierarchical source-discovery rewards (quick learning) vs long-term hiring
# reward (slow, high-value). Source bandit uses the quick tier.
SOURCE_REWARD_MAP: dict[str, float] = {
    "job_discovered": 0.01,
    "job_gated": 0.05,
    "job_exposed": 0.05,
    "high_fit": 0.20,
    "job_saved": 0.5,
    "job_applied": 1.0,
    "screening": 3.0,
    "interview": 8.0,
    "offer": 20.0,
}

# Discount factor for delayed rewards (reward * gamma**delta_days).
GAMMA = 0.99

# Separate funnel targets — never collapse to reward>0 (#2)
FUNNEL_STAGES = [
    "application_succeeds",
    "recruiter_response",
    "screening",
    "interview",
    "offer",
]

# Contextual bandit arms are separate policies — never one bandit controlling
# both recommendation and application. (#15, #16)
POLICY_TYPES = {
    "recommendation": "RecommendationPolicy",  # job -> recommend?
    "discovery": "DiscoveryPolicy",  # source/query -> crawl?
    "query": "QueryPolicy",  # query template -> issue?
    "board_routing": "BoardRoutingPolicy",  # board -> render/proxy?
    "application": "ApplicationPolicy",  # job -> apply? (gated)
}
