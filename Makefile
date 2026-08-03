# ho - one target per command type, flags pick the mode.
#
#   make run  MODE=overnight|no-cloud|worker|dev
#   make crawl MODE=crawl|ingest|notor
#   make intel TOP=N MODE=telegram|smart
#   make accepted OUTREACH=1 ELIG=all OUT=/tmp/x.csv
#   make fc ACTION=up|down|logs|status|clean|dev-down|tor-up
#   make graph ACTION=up|stop|reset|shell
#   make launch ACTION=start|stop|status|restart
#   make backup / backup-list / restore DIR=... / autobackup
#   make analytics / health / serve / check / test

.PHONY: help run crawl intel accepted analytics backup backup-list restore autobackup fc graph launch health serve check check-py check-node test

help:
	@echo "make <target> [FLAGS]"
	@echo ""
	@echo "  run            pipeline            MODE=(default|overnight|no-cloud|worker|dev)"
	@echo "  crawl          crawler + ingest   MODE=(default|crawl|ingest|notor)"
	@echo "  intel          recommendations    TOP=N  MODE=(telegram|smart|planning)"
	@echo "  accepted       dump candidates    OUTREACH=1 (outreach pack)  ELIG=(accepted|near_miss|rejected|all)  OUT=path"
	@echo "  analytics      system + DB stats"
	@echo "  backup / backup-list / restore    volume checkpoints (restore DIR=checkpoints/xxx)"
	@echo "  autobackup     snapshot + prune to 10"
	@echo "  fc             firecrawl stack    ACTION=(up|down|logs|status|clean|dev-down|tor-up)"
	@echo "  graph          neo4j              ACTION=(up|stop|reset|shell)"
	@echo "  launch         detached pipeline  ACTION=(start|stop|status|restart)"
	@echo "  health / serve / check / test"

# Pipeline
# MODE: (default) | no-cloud | worker | dev | overnight (direct orchestrator)
init-memory:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/autofill/scripts/init_memory.py

grill-persona:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/autofill/scripts/grill_persona.py

index-resume:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest uv run python packages/ingest/scripts/index_resume.py


run:
	@case "$(MODE)" in \
	  overnight) OVERNIGHT_LOOP=true PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python -m src.radar.engine.orchestrator ;; \
	  dev) PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/run.py --no-pipeline ;; \
	  worker) PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/run.py --worker-only ;; \
	  no-cloud) PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/run.py --no-cloud ;; \
	  *) PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/run.py ;; \
	esac

launch:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/cli/launch_pipeline.py --$(if $(ACTION),$(ACTION),start)

# Local crawler + ingest (via Tor)
# MODE: (default = crawl + ingest) | crawl | ingest | notor
crawl:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/cli/local_crawler.py $(if $(filter notor,$(MODE)),--no-tor,$(if $(MODE),--$(MODE)))

# Intel / exports
# TOP=N, MODE=telegram|smart
intel:
	@case "$(MODE)" in \
	  smart) PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/intel/smart_intel.py --write ;; \
	  planning) PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/intel/planning_pass.py --write ;; \
	  *) PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/intel/radar_intel.py $(if $(TOP),--top $(TOP)) $(if $(filter telegram,$(MODE)),--telegram) ;; \
	esac

# Dump accepted candidates. Default: apply-facing jobs columns.
#   make accepted              -> intel/accepted_jobs.csv
#   make accepted OUTREACH=1   -> intel/accepted_outreach.csv (founders/funding/socials/news)
#   make accepted ELIG=all     -> every eligibility
#   make accepted OUT=/tmp/x.csv
accepted:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/tools/export_accepted.py --out $(if $(OUT),$(OUT),intel/accepted_$(if $(OUTREACH),outreach,jobs).csv) --eligibility $(if $(ELIG),$(ELIG),accepted) --mode $(if $(OUTREACH),outreach,jobs)

analytics:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/tools/analytics.py

# Firecrawl containers + graph
# ACTION: up (default) | down | logs | status | clean | dev-down | tor-up
fc:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/cli/firecrawl.py $(if $(ACTION),$(ACTION),up)

# ACTION: up (default) | stop | reset | shell
graph:
	PYTHONPATH=$(CURDIR):$(CURDIR)/packages/ingest:$(CURDIR)/packages/autofill uv run python packages/ingest/scripts/cli/firecrawl.py graph-$(if $(ACTION),$(ACTION),up)

# Backups
backup:
	uv run python packages/ingest/scripts/backup/checkpoint_backup.py

backup-list:
	@ls -dt checkpoints/*/ 2>/dev/null || echo "no checkpoints yet"

restore:
	uv run python packages/ingest/scripts/backup/checkpoint_restore.py $(if $(DIR),--dir $(DIR))

autobackup:
	uv run python packages/ingest/scripts/backup/auto_backup.py

# Quality / infra
health:
	uv run python packages/ingest/scripts/tools/health.py

serve:
	uv run python packages/ingest/scripts/serve.py

check: check-py check-node

check-py:
	uv run ruff format . && uv run ruff check . --fix && uv run mypy packages/ingest/src packages/ingest/scripts packages/autofill --explicit-package-bases

check-node:
	cd packages/autofill/node && npm run lint && npm run format:check && npm run typecheck

test:
	uv run python -m pytest . -v --ignore=refs
