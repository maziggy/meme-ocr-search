#!/usr/bin/env python3
"""
Improved Meme OCR Text Extraction System

This version addresses common OCR issues:
- Better text cleaning and normalization
- Multiple OCR configurations
- Improved search with fuzzy matching
- Support for different languages
"""

import os
import json
import argparse
import re
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Any
import hashlib
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.table import Table

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR libraries not installed.")


def process_single_image(image_path_str: str) -> Dict:
    """
    Process a single image for OCR extraction.
    This function is designed to be called by worker processes.
    """
    if not OCR_AVAILABLE:
        return {
            "filepath": image_path_str,
            "filename": Path(image_path_str).name,
            "file_hash": "",
            "raw_text": "",
            "cleaned_text": "",
            "word_count": 0,
            "confidence": 0.0,
            "method": "no_ocr",
            "error": "OCR libraries not available"
        }
    
    try:
        image_path = Path(image_path_str)
        
        # Calculate file hash
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()
        
        # Extract text using multiple methods
        result = extract_text_multiple_methods_standalone(image_path_str)
        raw_text = result["text"]
        cleaned_text = clean_text_standalone(raw_text)
        confidence = result["confidence"]
        word_count = len(cleaned_text.split()) if cleaned_text else 0
        
        return {
            "filepath": image_path_str,
            "filename": image_path.name,
            "file_hash": file_hash,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "word_count": word_count,
            "confidence": confidence,
            "method": result["method"],
            "error": None
        }
        
    except Exception as e:
        return {
            "filepath": image_path_str,
            "filename": Path(image_path_str).name,
            "file_hash": "",
            "raw_text": "",
            "cleaned_text": "",
            "word_count": 0,
            "confidence": 0.0,
            "method": "error",
            "error": str(e)
        }


def clean_text_standalone(raw_text: str) -> str:
    """Standalone version of text cleaning for multiprocessing."""
    if not raw_text:
        return ""
    
    # Remove excessive whitespace and line breaks
    text = re.sub(r'\s+', ' ', raw_text.strip())
    
    # Fix common OCR errors
    replacements = {
        # Common character misreads
        '|': 'l',
        '1': 'l',  # in words context
        '0': 'o',  # in words context  
        '@': 'a',
        '5': 's',  # in words context
        '3': 'e',  # in words context
        
        # Remove noise characters
        r'[^\w\s\.,!?;:\'"()-]': '',
    }
    
    for pattern, replacement in replacements.items():
        if pattern.startswith('r'):
            text = re.sub(pattern[1:], replacement, text)
        else:
            text = text.replace(pattern, replacement)
    
    # Add spaces between likely word boundaries
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Clean up multiple spaces again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def score_text_quality_standalone(text: str) -> float:
    """Standalone version of text quality scoring."""
    if not text or not text.strip():
        return 0.0
    
    text = text.strip()
    
    # Base score from length
    score = min(len(text) / 50.0, 1.0) * 0.3
    
    # Character diversity bonus
    unique_chars = len(set(text.lower()))
    score += min(unique_chars / 20.0, 1.0) * 0.2
    
    # Word-like patterns bonus
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if 2 <= avg_word_len <= 8:  # Reasonable word length
            score += 0.2
    
    # Penalize too much noise
    noise_chars = len(re.findall(r'[^\w\s\.,!?;:\'"()-]', text))
    noise_ratio = noise_chars / max(len(text), 1)
    score -= noise_ratio * 0.3
    
    # Readable character ratio
    readable_chars = len(re.findall(r'[a-zA-Z0-9\s]', text))
    readable_ratio = readable_chars / max(len(text), 1)
    score += readable_ratio * 0.3
    
    return max(0.0, min(1.0, score))


def preprocess_image_standalone(image_path: str):
    """Standalone version of image preprocessing."""
    if not OCR_AVAILABLE:
        return None
        
    import cv2
    import numpy as np
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple preprocessing techniques
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Try adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh


def extract_text_multiple_methods_standalone(image_path: str) -> Dict:
    """Standalone version of OCR extraction for multiprocessing."""
    if not OCR_AVAILABLE:
        return {"text": "", "confidence": 0.0, "method": "none"}
    
    try:
        import pytesseract
        from PIL import Image
        import cv2
        import numpy as np
    except ImportError:
        return {"text": "", "confidence": 0.0, "method": "import_error"}
    
    results = []
    
    try:
        img = Image.open(image_path)
        
        # Method 1: Default configuration
        try:
            text1 = pytesseract.image_to_string(img)
            results.append({"text": text1, "method": "default"})
        except:
            pass
        
        # Method 2: PSM 6 (uniform block of text)
        try:
            text2 = pytesseract.image_to_string(img, config='--psm 6')
            results.append({"text": text2, "method": "psm6"})
        except:
            pass
        
        # Method 3: PSM 7 (single text line)
        try:
            text3 = pytesseract.image_to_string(img, config='--psm 7')
            results.append({"text": text3, "method": "psm7"})
        except:
            pass
        
        # Method 4: With preprocessing
        try:
            processed_img = preprocess_image_standalone(image_path)
            if processed_img is not None:
                # Convert back to PIL Image
                pil_img = Image.fromarray(processed_img)
                text4 = pytesseract.image_to_string(pil_img, config='--psm 6')
                results.append({"text": text4, "method": "preprocessed"})
        except:
            pass
        
        # Choose best result based on length and character diversity
        if not results:
            return {"text": "", "confidence": 0.0, "method": "failed"}
        
        best_result = max(results, key=lambda x: score_text_quality_standalone(x["text"]))
        best_result["confidence"] = score_text_quality_standalone(best_result["text"])
        
        return best_result
        
    except Exception as e:
        logger.error(f"Error extracting text from {image_path}: {e}")
        return {"text": "", "confidence": 0.0, "method": "error"}


class ImprovedMemeOCR:
    """Improved OCR with better text processing and search."""
    
    def __init__(self, memes_dir: str = "memes", db_path: str = "meme_index.db"):
        self.memes_dir = Path(memes_dir)
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create improved database with better text indexing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                filepath TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                raw_text TEXT,
                cleaned_text TEXT,
                word_count INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )
        ''')
        
        # Improved FTS with better tokenization
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS memes_fts USING fts5(
                filename, cleaned_text, 
                content='memes', 
                content_rowid='id',
                tokenize='porter ascii'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def clean_text(self, raw_text: str) -> str:
        """Clean and normalize extracted text."""
        if not raw_text:
            return ""
        
        # Remove excessive whitespace and line breaks
        text = re.sub(r'\s+', ' ', raw_text.strip())
        
        # Fix common OCR errors
        replacements = {
            # Common character misreads
            '|': 'l',
            '1': 'l',  # in words context
            '0': 'o',  # in words context  
            '@': 'a',
            '5': 's',  # in words context
            '3': 'e',  # in words context
            
            # Remove noise characters
            r'[^\w\s\.,!?;:\'"()-]': '',
        }
        
        for pattern, replacement in replacements.items():
            if pattern.startswith('r'):
                text = re.sub(pattern[1:], replacement, text)
            else:
                text = text.replace(pattern, replacement)
        
        # Add spaces between likely word boundaries
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Clean up multiple spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def extract_text_multiple_methods(self, image_path: str) -> Dict:
        """Try multiple OCR configurations and return the best result."""
        if not OCR_AVAILABLE:
            return {"text": "", "confidence": 0.0, "method": "none"}
        
        results = []
        
        try:
            img = Image.open(image_path)
            
            # Method 1: Default configuration
            try:
                text1 = pytesseract.image_to_string(img)
                results.append({"text": text1, "method": "default"})
            except:
                pass
            
            # Method 2: PSM 6 (uniform block of text)
            try:
                text2 = pytesseract.image_to_string(img, config='--psm 6')
                results.append({"text": text2, "method": "psm6"})
            except:
                pass
            
            # Method 3: PSM 7 (single text line)
            try:
                text3 = pytesseract.image_to_string(img, config='--psm 7')
                results.append({"text": text3, "method": "psm7"})
            except:
                pass
            
            # Method 4: PSM 8 (single word)
            try:
                text4 = pytesseract.image_to_string(img, config='--psm 8')
                results.append({"text": text4, "method": "psm8"})
            except:
                pass
            
            # Method 5: With preprocessing
            try:
                processed_img = self.preprocess_image(image_path)
                # Convert back to PIL Image
                pil_img = Image.fromarray(processed_img)
                text5 = pytesseract.image_to_string(pil_img, config='--psm 6')
                results.append({"text": text5, "method": "preprocessed"})
            except:
                pass
            
            # Choose best result based on length and character diversity
            if not results:
                return {"text": "", "confidence": 0.0, "method": "failed"}
            
            best_result = max(results, key=lambda x: self.score_text_quality(x["text"]))
            best_result["confidence"] = self.score_text_quality(best_result["text"])
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return {"text": "", "confidence": 0.0, "method": "error"}
    
    def score_text_quality(self, text: str) -> float:
        """Score text quality for choosing best OCR result."""
        if not text or not text.strip():
            return 0.0
        
        text = text.strip()
        
        # Base score from length
        score = min(len(text) / 50.0, 1.0) * 0.3
        
        # Character diversity bonus
        unique_chars = len(set(text.lower()))
        score += min(unique_chars / 20.0, 1.0) * 0.2
        
        # Word-like patterns bonus
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if 2 <= avg_word_len <= 8:  # Reasonable word length
                score += 0.2
        
        # Penalize too much noise
        noise_chars = len(re.findall(r'[^\w\s\.,!?;:\'"()-]', text))
        noise_ratio = noise_chars / max(len(text), 1)
        score -= noise_ratio * 0.3
        
        # Readable character ratio
        readable_chars = len(re.findall(r'[a-zA-Z0-9\s]', text))
        readable_ratio = readable_chars / max(len(text), 1)
        score += readable_ratio * 0.3
        
        return max(0.0, min(1.0, score))
    
    def preprocess_image(self, image_path: str):
        """Preprocess image for better OCR."""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Try adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def get_unprocessed_images(self):
        """Get list of images that haven't been processed yet."""
        if not self.memes_dir.exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all processed file hashes
        cursor.execute("SELECT file_hash FROM memes WHERE processed_at IS NOT NULL")
        processed_hashes = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        # Find unprocessed images
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        unprocessed = []
        
        for image_file in self.memes_dir.iterdir():
            if image_file.suffix.lower() not in supported_formats:
                continue
                
            file_hash = self.get_file_hash(image_file)
            if file_hash not in processed_hashes:
                unprocessed.append(image_file)
        
        return unprocessed
    
    def process_memes(self, force_reprocess: bool = False, verbose: bool = False, new_only: bool = False, num_workers: int = None):
        """Process memes with improved text extraction using parallel processing."""
        if not OCR_AVAILABLE:
            console = Console()
            console.print("[red]Cannot process memes: OCR libraries not installed[/red]")
            return
        
        if not self.memes_dir.exists():
            console = Console()
            console.print(f"[red]Memes directory '{self.memes_dir}' not found[/red]")
            return
        
        # Determine number of worker processes
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming
        
        # Determine which images to process
        if new_only and not force_reprocess:
            images_to_process = self.get_unprocessed_images()
        else:
            supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            images_to_process = [f for f in self.memes_dir.iterdir() 
                               if f.suffix.lower() in supported_formats]
        
        if not images_to_process:
            console = Console()
            if new_only:
                console.print("[green]No new images found to process![/green]")
            else:
                console.print("[yellow]No images found to process![/yellow]")
            return
        
        # Convert paths to strings for multiprocessing
        image_paths = [str(img) for img in images_to_process]
        
        # Initialize progress tracking
        console = Console()
        batch_size = 50  # Process in batches for better memory management
        total_batches = (len(image_paths) - 1) // batch_size + 1
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # Main progress bar
            main_task = progress.add_task(
                f"[cyan]Processing {len(image_paths)} images with {num_workers} workers...", 
                total=len(image_paths)
            )
            
            # Batch progress bar
            batch_task = progress.add_task(
                "[green]Batch progress...", 
                total=total_batches
            )
            
            # Current files being processed
            current_files_task = progress.add_task(
                "[yellow]Current files: Initializing...", 
                total=1, 
                visible=False
            )
            
            processed_count = 0
            error_count = 0
            start_time = time.time()
            
            # Process images in batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]
                
                progress.update(
                    batch_task, 
                    description=f"[green]Processing batch {batch_idx + 1}/{total_batches} ({len(batch_paths)} images)..."
                )
                
                # Track current processing files
                active_files = []
                
                # Process batch in parallel with real-time status
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all jobs
                    future_to_path = {executor.submit(process_single_image, path): path 
                                    for path in batch_paths}
                    
                    # Collect results as they complete
                    batch_results = []
                    completed_in_batch = 0
                    
                    for future in as_completed(future_to_path):
                        try:
                            result = future.result()
                            batch_results.append(result)
                            completed_in_batch += 1
                            
                            # Update current files display
                            remaining_futures = [f for f in future_to_path.keys() if not f.done()]
                            if remaining_futures:
                                current_files = [Path(future_to_path[f]).name for f in remaining_futures[:5]]
                                if len(remaining_futures) > 5:
                                    current_files.append(f"... +{len(remaining_futures) - 5} more")
                                
                                progress.update(
                                    current_files_task,
                                    description=f"[yellow]Processing: {', '.join(current_files)}",
                                    visible=True
                                )
                            else:
                                progress.update(current_files_task, visible=False)
                            
                            # Update main progress
                            if result["error"] is None:
                                processed_count += 1
                                if verbose:
                                    progress.console.print(
                                        f"[green]✓[/green] {result['filename']} - {result['method']} "
                                        f"(conf: {result['confidence']:.2f})"
                                    )
                            else:
                                error_count += 1
                                if verbose:
                                    progress.console.print(
                                        f"[red]✗[/red] {result['filename']} - ERROR: {result['error']}"
                                    )
                            
                            progress.update(main_task, advance=1)
                            
                        except Exception as e:
                            error_count += 1
                            progress.update(main_task, advance=1)
                            if verbose:
                                progress.console.print(f"[red]Failed to process {future_to_path[future]}: {e}[/red]")
                
                # Store batch results in database
                self._store_batch_results(batch_results)
                progress.update(batch_task, advance=1)
                
                # Update task descriptions with stats
                elapsed = time.time() - start_time
                avg_time = elapsed / max(processed_count + error_count, 1)
                progress.update(
                    main_task,
                    description=f"[cyan]Processed: {processed_count}, Errors: {error_count} "
                               f"({avg_time:.1f}s/img)"
                )
        
        # Final summary
        total_time = time.time() - start_time
        console.print("\n" + "="*60)
        console.print(f"[bold green]Processing Complete![/bold green]")
        console.print(f"[cyan]Total time:[/cyan] {total_time/60:.1f} minutes")
        console.print(f"[green]Successfully processed:[/green] {processed_count}")
        console.print(f"[red]Errors:[/red] {error_count}")
        console.print(f"[yellow]Average time per image:[/yellow] {total_time/(processed_count + error_count):.1f}s")
        console.print(f"[blue]Workers used:[/blue] {num_workers}")
        console.print("="*60)
    
    def _store_batch_results(self, results: List[Dict]):
        """Store a batch of processing results in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for result in results:
                if result["error"] is not None:
                    continue  # Skip failed results
                
                # Store in database
                cursor.execute('''
                    INSERT OR REPLACE INTO memes 
                    (filename, filepath, file_hash, raw_text, cleaned_text, 
                     word_count, confidence_score, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    result["filename"], 
                    result["filepath"], 
                    result["file_hash"], 
                    result["raw_text"],
                    result["cleaned_text"], 
                    result["word_count"], 
                    result["confidence"]
                ))
                
                # Update FTS index
                cursor.execute('''
                    INSERT OR REPLACE INTO memes_fts (rowid, filename, cleaned_text)
                    SELECT id, filename, cleaned_text FROM memes WHERE file_hash = ?
                ''', (result["file_hash"],))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing batch results: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def search_memes(self, query: str, limit: int = 20, min_confidence: float = 0.1) -> List[Dict]:
        """Enhanced search with better matching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        
        # Try exact FTS search first
        try:
            cursor.execute('''
                SELECT m.filename, m.filepath, m.cleaned_text, m.confidence_score,
                       snippet(memes_fts, 1, '<mark>', '</mark>', '...', 32) as snippet
                FROM memes_fts fts
                JOIN memes m ON m.id = fts.rowid
                WHERE memes_fts MATCH ? AND m.confidence_score >= ?
                ORDER BY rank
                LIMIT ?
            ''', (query, min_confidence, limit))
            
            for row in cursor.fetchall():
                results.append({
                    'filename': row[0],
                    'filepath': row[1],
                    'full_text': row[2],
                    'confidence': row[3],
                    'snippet': row[4],
                    'match_type': 'exact'
                })
        except Exception as e:
            logger.warning(f"FTS search failed: {e}")
        
        # If no results, try fuzzy search
        if not results:
            cursor.execute('''
                SELECT filename, filepath, cleaned_text, confidence_score
                FROM memes 
                WHERE cleaned_text LIKE ? AND confidence_score >= ?
                ORDER BY confidence_score DESC, word_count DESC
                LIMIT ?
            ''', (f'%{query}%', min_confidence, limit))
            
            for row in cursor.fetchall():
                results.append({
                    'filename': row[0],
                    'filepath': row[1],
                    'full_text': row[2],
                    'confidence': row[3],
                    'snippet': f"...{row[2]}...",
                    'match_type': 'fuzzy'
                })
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict:
        """Get enhanced statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memes")
        total_memes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM memes WHERE processed_at IS NOT NULL")
        processed_memes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM memes WHERE word_count > 0")
        memes_with_text = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence_score) FROM memes WHERE confidence_score > 0")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT COUNT(*) FROM memes WHERE confidence_score > 0.5")
        high_confidence = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_memes': total_memes,
            'processed_memes': processed_memes,
            'memes_with_text': memes_with_text,
            'avg_confidence': avg_confidence,
            'high_confidence_memes': high_confidence
        }
    
    def delete_meme(self, filepath: str) -> Dict[str, Any]:
        """Delete a meme from the database and optionally from disk."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First get the meme ID
            cursor.execute("SELECT id, filename FROM memes WHERE filepath = ?", (filepath,))
            result = cursor.fetchone()
            
            if not result:
                return {"success": False, "error": "Meme not found in database"}
            
            meme_id, filename = result
            
            # Delete from FTS index first
            cursor.execute("DELETE FROM memes_fts WHERE rowid = ?", (meme_id,))
            
            # Delete from main table
            cursor.execute("DELETE FROM memes WHERE id = ?", (meme_id,))
            
            # Commit the transaction
            conn.commit()
            
            # Try to delete the actual file
            file_deleted = False
            try:
                file_path = Path(filepath)
                if file_path.exists():
                    file_path.unlink()
                    file_deleted = True
            except Exception as e:
                logger.warning(f"Could not delete file {filepath}: {e}")
            
            return {
                "success": True,
                "filename": filename,
                "file_deleted": file_deleted,
                "message": f"Meme '{filename}' deleted from database" + (" and disk" if file_deleted else "")
            }
            
        except Exception as e:
            logger.error(f"Error deleting meme: {e}")
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description='Improved Meme OCR System')
    parser.add_argument('--extract', action='store_true', help='Extract text from all memes')
    parser.add_argument('--update', action='store_true', help='Process only new images that haven\'t been processed')
    parser.add_argument('--search', type=str, help='Search for memes containing text')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of all images')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--workers', '-w', type=int, help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--memes-dir', default='memes', help='Directory containing memes')
    parser.add_argument('--limit', type=int, default=20, help='Limit search results')
    parser.add_argument('--min-confidence', type=float, default=0.1, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    ocr = ImprovedMemeOCR(memes_dir=args.memes_dir)
    
    if args.extract:
        print("Extracting text with improved parallel OCR...")
        ocr.process_memes(force_reprocess=args.force, verbose=args.verbose, new_only=False, num_workers=args.workers)
        
    elif args.update:
        print("Processing new images with parallel OCR...")
        ocr.process_memes(force_reprocess=False, verbose=args.verbose, new_only=True, num_workers=args.workers)
        
    elif args.search:
        print(f"Searching for: '{args.search}'")
        results = ocr.search_memes(args.search, limit=args.limit, min_confidence=args.min_confidence)
        
        if not results:
            print("No memes found. Try:")
            print("- Lowering --min-confidence (default 0.1)")
            print("- Using simpler search terms")
            print("- Running --extract first to reprocess images")
        else:
            print(f"\nFound {len(results)} memes:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['filename']} (confidence: {result['confidence']:.2f}, {result['match_type']})")
                print(f"   Path: {result['filepath']}")
                if result['snippet']:
                    print(f"   Text: {result['snippet']}")
                else:
                    print(f"   Text: {result['full_text'][:100]}...")
    
    elif args.stats:
        stats = ocr.get_stats()
        print("\nImproved Meme Database Statistics:")
        print(f"Total memes: {stats['total_memes']}")
        print(f"Processed memes: {stats['processed_memes']}")
        print(f"Memes with text: {stats['memes_with_text']}")
        print(f"Average confidence: {stats['avg_confidence']:.2f}")
        print(f"High confidence memes (>0.5): {stats['high_confidence_memes']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    mp.set_start_method('spawn', force=True)
    main()