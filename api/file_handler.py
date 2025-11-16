"""
File upload and processing for CSV/Excel files.
"""

import pandas as pd
import re
import io
from typing import List, Optional, Dict, Any, Tuple
from fastapi import UploadFile, HTTPException
import logging

log = logging.getLogger("file_handler")


class FileProcessor:
    """Process uploaded CSV/Excel files for optimization and forecasting."""
    
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    # Company-provided dataset schema (canonical column names)
    SCHEMA_REQUIRED = [
        'Time stamp',
        'Inflow to tunnel F1',
    ]
    SCHEMA_OPTIONAL = [
        'Water level in tunnel L1',
        'Water volume in tunnel V',
        'Sum of pumped flow to WWTP F2',
        'Pump flow 1.1', 'Pump flow 1.2', 'Pump flow 1.3', 'Pump flow 1.4',
        'Pump flow 2.1', 'Pump flow 2.2', 'Pump flow 2.3', 'Pump flow 2.4',
        'pump power intake 1.1', 'pump power intake 1.2', 'pump power intake 1.3', 'pump power intake 1.4',
        'pump power intake 2.1', 'pump power intake 2.2', 'pump power intake 2.3', 'pump power intake 2.4',
        'Pump frequency 1.1', 'Pump frequency 1.2', 'Pump frequency 1.3', 'Pump frequency 1.4',
        'Pump frequency 2.1', 'Pump frequency 2.2', 'Pump frequency 2.3', 'Pump frequency 2.4',
        'Electricity price 1: high', 'Electricity price 2: normal'
    ]

    @staticmethod
    def validate_dataset_columns(columns: List[str]) -> Dict[str, Any]:
        """Validate columns against company schema.
        Returns dict with ok flag and missing columns lists.
        """
        cols_set = set(map(str, columns))
        missing_required = [c for c in FileProcessor.SCHEMA_REQUIRED if c not in cols_set]
        missing_optional = [c for c in FileProcessor.SCHEMA_OPTIONAL if c not in cols_set]
        return {
            "ok": len(missing_required) == 0,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
        }
    
    @staticmethod
    async def read_uploaded_file(
        file: UploadFile,
        skip_rows: Optional[List[int]] = None,
        drop_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Read and process uploaded CSV or Excel file.
        
        Args:
            file: Uploaded file object
            skip_rows: List of row indices to skip (0-based)
            drop_columns: List of column names to drop
            
        Returns:
            Processed pandas DataFrame
            
        Raises:
            HTTPException: If file format is invalid or processing fails
        """
        # Validate file extension
        filename = file.filename.lower()
        file_ext = None
        for ext in FileProcessor.SUPPORTED_FORMATS:
            if filename.endswith(ext):
                file_ext = ext
                break
        
        if not file_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {', '.join(FileProcessor.SUPPORTED_FORMATS)}"
            )
        
        # Read file content
        try:
            content = await file.read()
            
            # Check file size
            if len(content) > FileProcessor.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size: {FileProcessor.MAX_FILE_SIZE / 1024 / 1024} MB"
                )
            
            # Parse file based on format
            if file_ext == '.csv':
                df = pd.read_csv(io.BytesIO(content),skiprows=skip_rows if skip_rows else None)
            else:  # Excel formats
                df = pd.read_excel(io.BytesIO(content),skiprows=skip_rows if skip_rows else None)
            
            log.info(f"Loaded file '{file.filename}': {len(df)} rows, {len(df.columns)} columns")
            
            # Normalize column names: strip whitespace
            try:
                df.columns = df.columns.map(str).str.strip()
            except Exception:
                pass
            
            # # Skip specified rows
            # if skip_rows:
            #     valid_rows = [i for i in skip_rows if 0 <= i < len(df)]
            #     if valid_rows:
            #         df = df.drop(index=valid_rows).reset_index(drop=True)
            #         log.info(f"Skipped {len(valid_rows)} rows")
            
            # Drop specified columns
            if drop_columns:
                existing_cols = [col for col in drop_columns if col in df.columns]
                if existing_cols:
                    df = df.drop(columns=existing_cols)
                    log.info(f"Dropped columns: {existing_cols}")
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            return df
            
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="File is empty")
        except pd.errors.ParserError as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
        except Exception as e:
            log.error(f"Error processing file: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    @staticmethod
    def extract_inflow_data(
        df: pd.DataFrame,
        inflow_column: str = "Inflow to tunnel F1",
        timestamp_column: Optional[str] = "Time stamp"
    ) -> Tuple[List[float], Optional[List[Any]], str]:
        """
        Extract inflow data from DataFrame.
        
        Args:
            df: Processed DataFrame
            inflow_column: Name of column containing inflow data
            timestamp_column: Optional name of column containing timestamps
            
        Returns:
            Tuple of (inflow_values, timestamps)
            
        Raises:
            HTTPException: If required column not found
        """
        # Resolve inflow column with tolerant matching
        resolved_col = inflow_column
        columns = list(df.columns)
        if inflow_column not in columns:
            # Case-insensitive exact
            ci_matches = [col for col in columns if str(col).lower() == inflow_column.lower()]
            if ci_matches:
                resolved_col = ci_matches[0]
            else:
                # Fuzzy contains: match tokens
                def normalize(s: str) -> List[str]:
                    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]
                q_tokens = normalize(inflow_column)
                scored: List[tuple[int, str]] = []
                for col in columns:
                    tokens = normalize(str(col))
                    score = 0
                    # token overlap
                    score += sum(1 for t in q_tokens if t in tokens)
                    # common synonyms boost
                    if "inflow" in tokens or "flow" in tokens:
                        score += 1
                    if "tunnel" in tokens:
                        score += 1
                    if "f1" in tokens:
                        score += 1
                    # direct substring bonus
                    if inflow_column.lower() in str(col).lower():
                        score += 1
                    if score:
                        scored.append((score, col))
                if scored:
                    scored.sort(key=lambda x: (-x[0], x[1]))
                    # pick top unique candidate if highest score is unique
                    top_score = scored[0][0]
                    top_candidates = [c for s, c in scored if s == top_score]
                    if len(top_candidates) == 1:
                        resolved_col = top_candidates[0]
                    else:
                        # Ambiguous
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Ambiguous inflow_column '{inflow_column}'. Candidates: {top_candidates}. "
                                f"Please set 'inflow_column' to one of them."
                            )
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column '{inflow_column}' not found. Available: {columns}"
                    )
        
        # Extract inflow values
        try:
            inflow_values = df[resolved_col].dropna().tolist()
            
            # Convert to float
            inflow_values = [float(val) for val in inflow_values]
            
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert '{resolved_col}' to numeric values: {str(e)}"
            )
        
        # Extract timestamps if specified
        timestamps = None
        if timestamp_column:
            if timestamp_column not in df.columns:
                matching_cols = [col for col in df.columns if col.lower() == timestamp_column.lower()]
                if matching_cols:
                    timestamp_column = matching_cols[0]
                else:
                    log.warning(f"Timestamp column '{timestamp_column}' not found")
                    return inflow_values, None, resolved_col
            
            try:
                timestamps = pd.to_datetime(df[timestamp_column]).dropna().tolist()
            except Exception as e:
                log.warning(f"Failed to parse timestamps: {e}")
        
        return inflow_values, timestamps, resolved_col
    
    @staticmethod
    def get_data_preview(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get preview of first n rows as list of dictionaries.
        
        Args:
            df: DataFrame to preview
            n: Number of rows to include
            
        Returns:
            List of row dictionaries
        """
        preview_df = df.head(n)
        return preview_df.to_dict('records')
    
    @staticmethod
    def validate_inflow_data(inflow_values: List[float], min_length: int = 32) -> None:
        """
        Validate inflow data meets requirements.
        
        Args:
            inflow_values: List of inflow values
            min_length: Minimum required data points
            
        Raises:
            HTTPException: If validation fails
        """
        if len(inflow_values) < min_length:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Need at least {min_length} points, got {len(inflow_values)}"
            )
        
        if any(val < 0 for val in inflow_values):
            raise HTTPException(
                status_code=400,
                detail="Inflow values must be non-negative"
            )
        
        # Check for too many zeros or constant values
        unique_values = len(set(inflow_values))
        if unique_values < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Data has too few unique values ({unique_values}). Need more variation."
            )
