# VLMnav — Zero-shot ObjectNav with Cascaded YOLO+VLM Detection and Depth-based 2D Mapping

> Work in progress. Zero-shot object-goal navigation on AVDB scenes in habitat_sim:
> semantic tier (Qwen-VL, present-only) + metric tier (depth → persistent 4-state occupancy grid,
> stop-gate with geometric evidence audit). See `config/`, `src/`, `bm_run.py`, `test_*.py`.
>
> Setup: `cp .env.example .env` (fill `QWEN_API_KEY`) → see `requirements.txt` (habitat-sim 0.3.1, headless).

*(Full documentation will be added before release.)*
