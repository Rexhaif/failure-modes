#!/usr/bin/env python3
"""
Script to add error indicators to evaluation datasets.

This script processes MT and TS evaluation datasets and adds a binary error indicator
column based on the difference between LLM scores and ground truth scores.

For MT datasets: Error when |llm_mqm_score - golden_mqm_score| > 5
For TS datasets: Error when normalized_llm_score != normalized_true_score

Before saving, removes previously used error columns and backtracking data.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


console = Console()


def setup_logging() -> None:
    """Configure rich console for colored output."""
    console.rule("[bold blue]Error Indicator Generator", style="blue")


def validate_file_path(file_path: str) -> Path:
    """Validate that the file exists and is a CSV file."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File '{file_path}' not found")
        sys.exit(1)
    if path.suffix.lower() != '.csv':
        console.print(f"[bold red]Error:[/bold red] File '{file_path}' is not a CSV file")
        sys.exit(1)
    return path


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Detect whether this is an MT or TS dataset based on column names."""
    mt_columns = {'golden_mqm_score', 'llm_mqm_score'}
    ts_columns = {'normalized_llm_score', 'normalized_true_score'}
    
    if mt_columns.issubset(df.columns):
        return 'mt'
    elif ts_columns.issubset(df.columns):
        return 'ts'
    else:
        console.print(f"[bold red]Error:[/bold red] Cannot detect dataset type")
        console.print(f"Expected columns for MT: {mt_columns}")
        console.print(f"Expected columns for TS: {ts_columns}")
        console.print(f"Found columns: {set(df.columns)}")
        sys.exit(1)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove previously used error columns and backtracking data."""
    columns_to_remove = []
    
    existing_columns_to_remove = [
        'golden_mqm_score', 'llm_mqm_score',
        'normalized_true_score', 'normalized_llm_score',
        'normalized_error', 'backtracking_count',
        'backtracking_patterns'
    ]

    console.print(f"[yellow]Removing columns:[/yellow] {existing_columns_to_remove}")
    for existing_column in existing_columns_to_remove:
        if existing_column in df.columns:
            df = df.drop(columns=[existing_column])
    
    return df


def add_mt_error_indicator(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Add error indicator for MT dataset based on MQM score difference."""
    console.print(f"[yellow]Processing MT dataset with threshold: {threshold}[/yellow]")
    
    # Calculate absolute difference
    df['score_diff'] = abs(df['llm_mqm_score'] - df['golden_mqm_score'])
    
    # Add error indicator (1 if error, 0 if not)
    df['error_indicator'] = (df['score_diff'] > threshold).astype(int)
    
    # Remove temporary column
    df = df.drop(columns=['score_diff'])
    
    return df


def add_ts_error_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add error indicator for TS dataset based on any score difference."""
    console.print("[yellow]Processing TS dataset (any difference is error)[/yellow]")
    
    # Add error indicator (1 if error, 0 if not)
    df['error_indicator'] = (df['normalized_llm_score'] != df['normalized_true_score']).astype(int)
    
    return df


def display_statistics(df: pd.DataFrame, dataset_type: str) -> None:
    """Display error statistics in a formatted table."""
    total_rows = len(df)
    error_count = df['error_indicator'].sum()
    accuracy = (total_rows - error_count) / total_rows * 100
    
    # Create statistics table
    table = Table(title=f"Error Statistics for {dataset_type.upper()} Dataset")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Percentage", style="green")
    
    table.add_row("Total Rows", str(total_rows), "100.0%")
    table.add_row("Errors", str(error_count), f"{error_count/total_rows*100:.1f}%")
    table.add_row("Correct", str(total_rows - error_count), f"{accuracy:.1f}%")
    
    console.print(table)


def process_dataset(
    input_path: Path, 
    output_path: Optional[Path] = None, 
    threshold: float = 5.0
) -> Tuple[pd.DataFrame, str]:
    """Process a single dataset file."""
    console.print(f"[blue]Loading dataset:[/blue] {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        console.print(f"[green]✓[/green] Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        console.print(f"[bold red]Error loading CSV:[/bold red] {e}")
        sys.exit(1)
    
    # Detect dataset type
    dataset_type = detect_dataset_type(df)
    console.print(f"[blue]Detected dataset type:[/blue] {dataset_type.upper()}")
    
    # Add error indicators based on dataset type
    if dataset_type == 'mt':
        df = add_mt_error_indicator(df, threshold)
    else:  # ts
        df = add_ts_error_indicator(df)
    
    # Display statistics
    display_statistics(df, dataset_type)
    
    # Save output
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_with_errors{input_path.suffix}"
    
    df = clean_dataframe(df)
    try:
        df.to_csv(output_path, index=False)
        console.print(f"[green]✓[/green] Output saved to: {output_path}")
    except Exception as e:
        console.print(f"[bold red]Error saving output:[/bold red] {e}")
        sys.exit(1)
    
    return df, dataset_type


def main() -> None:
    """Main function to handle command line arguments and process datasets."""
    parser = argparse.ArgumentParser(
        description="Add error indicators to evaluation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process MT dataset with default threshold (5.0)
  python src/failure-modes/add_error_indicators.py data/mt@deepseek_r1.csv

  # Process TS dataset
  python src/failure-modes/add_error_indicators.py data/ts@deepseek_r1.csv

  # Process with custom threshold and output path
  python src/failure-modes/add_error_indicators.py data/mt@deepseek_r1.csv -t 3.0 -o output.csv

  # Process both datasets
  python src/failure-modes/add_error_indicators.py data/mt@deepseek_r1.csv data/ts@deepseek_r1.csv
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input CSV file(s) to process'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: input_file_with_errors.csv)'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=5.0,
        help='Threshold for MT datasets (default: 5.0)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output file if it exists'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Validate input files
    input_paths = [validate_file_path(f) for f in args.input_files]
    
    # Check if multiple files but single output specified
    if len(input_paths) > 1 and args.output:
        console.print("[bold red]Error:[/bold red] Cannot specify single output for multiple input files")
        sys.exit(1)
    
    # Process each dataset
    with Progress() as progress:
        task = progress.add_task("Processing datasets...", total=len(input_paths))
        
        for i, input_path in enumerate(input_paths):
            console.print(f"\n[bold blue]Processing file {i+1}/{len(input_paths)}:[/bold blue]")
            
            output_path = None
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not args.overwrite:
                    console.print(f"[bold red]Error:[/bold red] Output file '{output_path}' exists. Use --overwrite to replace it.")
                    sys.exit(1)
            
            process_dataset(input_path, output_path, args.threshold)
            progress.update(task, advance=1)
    
    console.print("\n[bold green]✓ All datasets processed successfully![/bold green]")


if __name__ == "__main__":
    main()