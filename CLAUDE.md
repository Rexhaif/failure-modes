# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for analyzing failure modes in AI systems, specifically focused on translation evaluation and reasoning quality assessment. The project contains:

- CSV datasets with translation evaluation data (MQM scores, reasoning content, backtracking patterns)
- Two main data files: `mt@deepseek_r1.csv` (machine translation evaluations) and `ts@deepseek_r1.csv` (text summarization evaluations)

## Development Commands

The project uses UV for package management with `pyproject.toml`:

```bash
# Sync dependencies with UV
uv sync

# Run the main application
python main.py

# Add error indicators to datasets
python src/failure-modes/add_error_indicators.py data/mt@deepseek_r1.csv data/ts@deepseek_r1.csv

# Process specific dataset with custom threshold
python src/failure-modes/add_error_indicators.py data/mt@deepseek_r1.csv -t 3.0 -o custom_output.csv
```

## Code Architecture

- `main.py`: Entry point with basic hello world functionality
- `src/failure-modes/add_error_indicators.py`: Script to add binary error indicators to evaluation datasets
- `data/`: Contains CSV files with evaluation data
  - `mt@deepseek_r1.csv`: Machine translation evaluation data with columns for system, source/hypothesis segments, language pairs, MQM scores, reasoning tokens, and backtracking patterns
  - `ts@deepseek_r1.csv`: Text summarization evaluation data with normalized scores and reasoning traces
  - `*_with_errors.csv`: Enhanced datasets with error indicator columns
- `src/failure-modes/`: Empty source directory structure

## Data Structure

The CSV files contain evaluation data with:
- Translation quality metrics (MQM scores)
- LLM reasoning content and token counts
- Backtracking patterns and counts
- Normalized scoring for comparison analysis

## Error Detection Logic

The `add_error_indicators.py` script implements dataset-specific error detection:

### MT Dataset Error Detection
- **Threshold**: |llm_mqm_score - golden_mqm_score| > 5.0
- **Current Error Rate**: 54.7% (4,264/7,798 rows)
- **Columns**: `golden_mqm_score`, `llm_mqm_score`, `error_indicator`

### TS Dataset Error Detection
- **Threshold**: Any difference between normalized scores
- **Current Error Rate**: 81.6% (5,220/6,397 rows)  
- **Columns**: `normalized_llm_score`, `normalized_true_score`, `error_indicator`

## Dependencies

- `pandas>=1.5.0`: Data manipulation and CSV processing
- `rich>=13.0.0`: Colorful console output and progress visualization