# Changelog

## [1.0.0] — 2026-03-29

### Added
- Complete OpenEnv-compliant environment for Contract Clause Review
- 3 difficulty-tiered tasks (easy, medium, hard)
- 9 scenario files (3 per task) with realistic legal clauses
- 15 clause types, 4 risk levels, 13 issue flags, 5 suggested actions
- Per-step reward functions with partial credit scoring
- Deterministic trajectory-level graders (0.0–1.0)
- FastAPI server on port 7860 with CORS support
- LLM inference script with 4-level JSON parsing fallback
- Dual-mode operation (direct in-process + HTTP)
- Production Dockerfile for HF Spaces
- Comprehensive test suite (phases 1–10)
- Grader calibration verification
- Adversarial stress test suite
- Multi-scenario benchmark runner
- Multi-model evaluation framework
- Pre-submission validation scripts
- GitHub Actions CI/CD pipeline
- Post-deployment health monitor
