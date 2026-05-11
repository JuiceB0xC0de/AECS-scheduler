# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of AECS (Adaptive Event-Control Scheduler)
- Four operating modes: BASELINE, RECOVERY, EXPLORE, STABILIZE
- Event detection via gradient z-score, loss spikes, redundancy, and instability
- HuggingFace Trainer callback support
- Configurable thresholds and tuning parameters

### Features
- State-aware, event-driven learning rate scheduling
- Automatic mode switching based on training signals
- LR modulation with cosine backbone
- Momentum and weight decay tweaks per mode

## Unreleased

### Planned
- Unit test coverage
- Additional benchmark results
- Documentation improvements
