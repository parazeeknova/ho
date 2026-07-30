"""Company-specific ATS board registry.

Generic vendor roots (boards.greenhouse.io) are never mapped.
Each entry's URL is an actual company board that returns direct postings.
"""

_BOARDS_DATA = [
    ("openai:greenhouse", "https://boards.greenhouse.io/openai", "greenhouse"),
    ("anthropic:ashby", "https://jobs.ashbyhq.com/anthropic", "ashby"),
    ("stripe:greenhouse", "https://boards.greenhouse.io/stripe", "greenhouse"),
    ("airbnb:greenhouse", "https://boards.greenhouse.io/airbnb", "greenhouse"),
    ("databricks:greenhouse", "https://boards.greenhouse.io/databricks", "greenhouse"),
    ("figma:greenhouse", "https://boards.greenhouse.io/figma", "greenhouse"),
    ("notion:greenhouse", "https://boards.greenhouse.io/notion", "greenhouse"),
    ("vercel:greenhouse", "https://boards.greenhouse.io/vercel", "greenhouse"),
    ("cloudflare:greenhouse", "https://boards.greenhouse.io/cloudflare", "greenhouse"),
    ("mongodb:greenhouse", "https://boards.greenhouse.io/mongodb", "greenhouse"),
    ("snowflake:greenhouse", "https://boards.greenhouse.io/snowflake", "greenhouse"),
    ("doordash:greenhouse", "https://boards.greenhouse.io/doordash", "greenhouse"),
    ("reddit:greenhouse", "https://boards.greenhouse.io/reddit", "greenhouse"),
    ("roblox:greenhouse", "https://boards.greenhouse.io/roblox", "greenhouse"),
    ("spotify:greenhouse", "https://boards.greenhouse.io/spotify", "greenhouse"),
    ("duolingo:greenhouse", "https://boards.greenhouse.io/duolingo", "greenhouse"),
    ("twilio:greenhouse", "https://boards.greenhouse.io/twilio", "greenhouse"),
    ("gitlab:greenhouse", "https://boards.greenhouse.io/gitlab", "greenhouse"),
    ("hashicorp:greenhouse", "https://boards.greenhouse.io/hashicorp", "greenhouse"),
    ("confluent:greenhouse", "https://boards.greenhouse.io/confluent", "greenhouse"),
    ("postman:lever", "https://jobs.lever.co/postman", "lever"),
    ("browserstack:lever", "https://jobs.lever.co/browserstack", "lever"),
    ("hasura:lever", "https://jobs.lever.co/hasura", "lever"),
    ("razorpay:lever", "https://jobs.lever.co/razorpay", "lever"),
    ("cohere:lever", "https://jobs.lever.co/cohere", "lever"),
    ("pinecone:lever", "https://jobs.lever.co/pinecone", "lever"),
    ("scaleai:ashby", "https://jobs.ashbyhq.com/scaleai", "ashby"),
    ("perplexity:ashby", "https://jobs.ashbyhq.com/perplexity", "ashby"),
    ("character:ashby", "https://jobs.ashbyhq.com/character", "ashby"),
    ("anyscale:ashby", "https://jobs.ashbyhq.com/anyscale", "ashby"),
    ("weaviate:ashby", "https://jobs.ashbyhq.com/weaviate", "ashby"),
    ("wandb:ashby", "https://jobs.ashbyhq.com/wandb", "ashby"),
    ("replicate:ashby", "https://jobs.ashbyhq.com/replicate", "ashby"),
    ("together:ashby", "https://jobs.ashbyhq.com/together", "ashby"),
    ("huggingface:workable", "https://apply.workable.com/huggingface", "workable"),
    ("ycombinator:jobs", "https://www.ycombinator.com/jobs", "careers_page"),
    ("wellfound:jobs", "https://wellfound.com/jobs", "careers_page"),
]


def get_company_boards() -> list[dict[str, str]]:
    return [{"id": id_, "url": url, "adapter": adapter} for id_, url, adapter in _BOARDS_DATA]
