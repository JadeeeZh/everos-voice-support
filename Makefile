RUN ?= v1
SRC ?= v1
MIDCALL ?= 8

.PHONY: setup smoke learn evaluate report clean relearn reuse-baseline evaluate-everos

setup:
	bash scripts/setup.sh

# End-to-end sanity: 3 learning calls + 2 eval calls per arm, tiny scale.
smoke:
	uv run python -m voicedemo.learn --run smoke --limit 3
	uv run python -m voicedemo.evaluate --run smoke --limit 2
	uv run python -m voicedemo.report --run smoke

learn:
	uv run python -m voicedemo.learn --run $(RUN)

evaluate:
	uv run python -m voicedemo.evaluate --run $(RUN)

report:
	uv run python -m voicedemo.report --run $(RUN)

# v2 recipe (PLAN.md "v2 levers"). Full flow:
#   make relearn SRC=v1 RUN=v2       # replay v1 call logs, v2 consolidation prompt
#   make reuse-baseline SRC=v1 RUN=v2
#   make evaluate-everos RUN=v2      # everos arm only, mid-call retrieval on
#   make report RUN=v2
relearn:
	uv run python -m voicedemo.learn --run $(RUN) --replay-from $(SRC) --skill-prompt v2

reuse-baseline:
	mkdir -p runs/$(RUN)/eval/sims
	cp runs/$(SRC)/eval/baseline.jsonl runs/$(RUN)/eval/
	cp runs/$(SRC)/eval/sims/baseline-*.json runs/$(RUN)/eval/sims/

evaluate-everos:
	uv run python -m voicedemo.evaluate --run $(RUN) --arms everos --midcall-turn $(MIDCALL)

clean:
	rm -rf runs/smoke
