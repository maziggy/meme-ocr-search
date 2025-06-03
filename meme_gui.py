#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio Web GUI for Meme OCR System

A modern web interface for the meme text extraction and search system.
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import os
import sqlite3
from typing import List, Tuple, Optional, Dict, Any
import multiprocessing as mp
from PIL import Image
import threading
import time
import json
import tempfile
import shutil

# Import our OCR system
from meme_ocr import ImprovedMemeOCR, process_single_image, OCR_AVAILABLE

class MemeGUI:
    def __init__(self):
        self.ocr = ImprovedMemeOCR()
        self.processing_active = False
        self.app_dir = Path(__file__).parent.absolute()
        self.default_memes_path = str(self.app_dir / "memes")
        
    def get_stats(self):
        """Get current database statistics."""
        stats = self.ocr.get_stats()
        return f"""
        <div style="
            width: 100%;
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        ">
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div><strong>Database Statistics</strong></div>
                <div><strong>Total memes:</strong> {stats['total_memes']}</div>
                <div><strong>Processed memes:</strong> {stats['processed_memes']}</div>
                <div><strong>Memes with text:</strong> {stats['memes_with_text']}</div>
                <div><strong>Average confidence:</strong> {stats['avg_confidence']:.2f}</div>
                <div><strong>High confidence (>0.5):</strong> {stats['high_confidence_memes']}</div>
            </div>
        </div>
        """
    
    def search_memes(self, query: str, max_results: int = 20, min_confidence: float = 0.1, preview_size: int = 250):
        """Search for memes and return results with images."""
        if not query.strip():
            return gr.update(visible=False), "Enter a search term to find memes."
        
        results = self.ocr.search_memes(query, limit=max_results, min_confidence=min_confidence)
        
        if not results:
            return gr.update(visible=False), f"No memes found for '{query}'. Try lowering the confidence threshold or different search terms."
        
        # Format results for gallery
        gallery_items = []
        info_text = f"Found {len(results)} memes matching '{query}':\n\n"
        
        # Create temp directory for Gradio-safe file serving
        temp_dir = tempfile.mkdtemp()
        
        for i, result in enumerate(results, 1):
            image_path = result['filepath']
            if os.path.exists(image_path):
                try:
                    # Load and resize image for preview
                    from PIL import Image
                    img = Image.open(image_path)
                    
                    # Calculate new size maintaining aspect ratio
                    width, height = img.size
                    aspect_ratio = width / height
                    
                    if width > height:
                        new_width = preview_size
                        new_height = int(preview_size / aspect_ratio)
                    else:
                        new_height = preview_size
                        new_width = int(preview_size * aspect_ratio)
                    
                    # Resize image
                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save resized image to temp directory
                    temp_image_path = os.path.join(temp_dir, f"result_{i}_{result['filename']}")
                    img_resized.save(temp_image_path)
                    
                    # Create caption with extracted text
                    caption = f"{result['filename']}\n"
                    caption += f"Confidence: {result['confidence']:.2f}\n"
                    if result.get('snippet'):
                        caption += f"Text: {result['snippet']}"
                    else:
                        text_preview = result['full_text'][:100]
                        if len(result['full_text']) > 100:
                            text_preview += "..."
                        caption += f"Text: {text_preview}"
                    
                    gallery_items.append((temp_image_path, caption))
                    
                    # Add to info text
                    info_text += f"{i}. **{result['filename']}** (conf: {result['confidence']:.2f})\n"
                    if result.get('snippet'):
                        info_text += f"   Text: {result['snippet']}\n"
                    info_text += "\n"
                except Exception as e:
                    print(f"Error copying image {result['filename']}: {e}")
                    continue
        
        # Determine optimal columns based on preview size
        columns_map = {100: 8, 150: 6, 200: 5, 250: 4, 300: 4, 350: 3, 400: 3, 450: 2, 500: 2}
        columns = columns_map.get(preview_size, 4)
        
        # Create HTML grid with improved context menu
        import base64
        
        # Create HTML grid with context menu - use simple onclick approach
        html_grid = f"""
        <div style="margin-top: 20px; width: 100%; overflow: hidden;">
            <div id="image-grid" style="
                display: grid;
                grid-template-columns: repeat({columns}, 1fr);
                gap: 15px;
                padding: 15px;
                width: 100%;
                box-sizing: border-box;
                overflow: visible;
            ">
        """
        
        for i, (img_path, caption) in enumerate(gallery_items):
            # Read image and convert to base64
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            # Get original image path for copying
            orig_path = results[i]['filepath']
            
            # Read original image for copying
            with open(orig_path, 'rb') as f:
                orig_img_data = base64.b64encode(f.read()).decode()
            
            html_grid += f"""
            <div style="position: relative; background: #f8f8f8; border-radius: 8px; overflow: visible; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="overflow: hidden; border-radius: 8px;">
                    <img src="data:image/jpeg;base64,{img_data}" 
                         data-original="data:image/jpeg;base64,{orig_img_data}"
                         data-filepath="{orig_path}"
                         style="width: 100%; height: auto; display: block; cursor: pointer;"
                         title="{caption.replace('"', '&quot;')}"
                         onclick="window.open(this.getAttribute('data-original'), '_blank', 'width=' + window.innerWidth * 0.8 + ',height=' + window.innerHeight * 0.8 + ',resizable=yes,scrollbars=yes')" />
                </div>
                
                <!-- Menu Button -->
                <button class="menu-btn" id="menu-btn-{i}"
                        onclick="window.toggleMenu({i}, event); return false;"
                        style="
                            position: absolute;
                            top: 8px;
                            left: 8px;
                            background: rgba(0, 0, 0, 0.75);
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            border-radius: 6px;
                            padding: 8px;
                            cursor: pointer;
                            color: white;
                            font-size: 16px;
                            line-height: 1;
                            transition: all 0.2s;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            width: 32px;
                            height: 32px;
                        "
                        onmouseover="this.style.background='rgba(0, 0, 0, 0.9)'"
                        onmouseout="this.style.background='rgba(0, 0, 0, 0.75)'">
                    ⋮
                </button>
                
                <!-- Context Menu -->
                <div id="menu-{i}" style="
                    position: absolute;
                    top: 42px;
                    left: 8px;
                    background: #ffffff;
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    padding: 8px;
                    display: none;
                    min-width: 160px;
                    z-index: 50000;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                ">
                    <div onclick="window.copyImage({i})" style="
                        padding: 12px 16px;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        border-radius: 6px;
                        transition: background-color 0.2s;
                        color: #333;
                        font-size: 14px;
                        font-weight: 500;
                    " onmouseover="this.style.backgroundColor='#f5f5f5'" onmouseout="this.style.backgroundColor='transparent'">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: #2196F3;">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span>Copy Image</span>
                    </div>
                    
                    <div onclick="window.deleteImage({i}, '{orig_path}')" style="
                        padding: 12px 16px;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        border-radius: 6px;
                        transition: background-color 0.2s;
                        color: #d32f2f;
                        font-size: 14px;
                        font-weight: 500;
                    " onmouseover="this.style.backgroundColor='#ffebee'" onmouseout="this.style.backgroundColor='transparent'">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: #f44336;">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            <line x1="10" y1="11" x2="10" y2="17"></line>
                            <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                        <span>Delete</span>
                    </div>
                </div>
                
                <div style="padding: 10px; font-size: 12px; line-height: 1.4;">
                    {caption.replace(chr(10), '<br>')}
                </div>
            </div>
            """
        
        html_grid += """
            </div>
        </div>
        """
        
        return gr.update(value=html_grid, visible=True), info_text
    
    def delete_meme(self, filepath: str) -> Dict[str, Any]:
        """Delete a meme from the database and disk."""
        try:
            result = self.ocr.delete_meme(filepath)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_folder(self, folder_path: str, num_workers: int = 10, progress=gr.Progress()):
        """Process all images in a folder."""
        if not folder_path or not os.path.exists(folder_path):
            return "ERROR Invalid folder path"
        
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            return "ERROR Path is not a directory"
        
        # Find image files
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in folder_path.rglob('*') 
                      if f.suffix.lower() in supported_formats]
        
        if not image_files:
            return "ERROR No image files found in the folder"
        
        self.processing_active = True
        processed_count = 0
        error_count = 0
        start_time = time.time()
        
        try:
            # Initialize status
            yield f"Starting **Starting Processing**\n\n* Found {len(image_files)} images\n* Using {num_workers} workers\n* Initializing parallel processing...", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), self.get_stats()
            
            # Process in batches for better memory management
            batch_size = 50
            total_batches = (len(image_files) - 1) // batch_size + 1
            
            for batch_idx in range(total_batches):
                if not self.processing_active:
                    yield f"STOP **Processing Stopped**\n\n* Processed: {processed_count}\n* Errors: {error_count}\n* Status: Stopped by user", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), self.get_stats()
                    return
                
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(image_files))
                batch_files = image_files[start_idx:end_idx]
                
                # Update status for current batch
                elapsed = time.time() - start_time
                avg_time = elapsed / max(processed_count + error_count, 1)
                remaining = len(image_files) - (processed_count + error_count)
                eta = avg_time * remaining
                
                status = f"Processing **Processing Batch {batch_idx + 1}/{total_batches}**\n\n"
                status += f"* **Current batch:** {len(batch_files)} images\n"
                status += f"* **Progress:** {processed_count + error_count}/{len(image_files)} images\n"
                status += f"* **Completed:** {processed_count} SUCCESS\n"
                status += f"* **Errors:** {error_count} ERROR\n"
                status += f"* **Elapsed time:** {elapsed/60:.1f} minutes\n"
                status += f"* **Avg time/image:** {avg_time:.1f}s\n"
                if remaining > 0:
                    status += f"* **ETA:** {eta/60:.1f} minutes\n"
                status += f"* **Workers active:** {num_workers}\n\n"
                
                # Process batch in parallel
                batch_results = []
                
                if num_workers > 1:
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        future_to_path = {executor.submit(process_single_image, str(path)): path 
                                        for path in batch_files}
                        
                        # Initial status with all submitted files
                        status += f"Refresh **Currently Processing:**\n"
                        all_files = [Path(str(f)).name for f in batch_files]
                        display_files = all_files[:5]
                        if len(all_files) > 5:
                            display_files.append(f"... +{len(all_files) - 5} more")
                        for i, filename in enumerate(display_files, 1):
                            if not filename.startswith("..."):
                                status += f"   {i}. `{filename}`\n"
                            else:
                                status += f"   {filename}\n"
                        
                        yield status, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), self.get_stats()
                        
                        for future in as_completed(future_to_path):
                            try:
                                result = future.result()
                                batch_results.append(result)
                                
                                if result["error"] is None:
                                    processed_count += 1
                                else:
                                    error_count += 1
                                
                                # Update status with remaining active processes
                                remaining_futures = [f for f in future_to_path.keys() if not f.done()]
                                if remaining_futures:
                                    elapsed = time.time() - start_time
                                    avg_time = elapsed / max(processed_count + error_count, 1)
                                    remaining_total = len(image_files) - (processed_count + error_count)
                                    eta = avg_time * remaining_total
                                    
                                    current_status = f"Processing **Processing Batch {batch_idx + 1}/{total_batches}**\n\n"
                                    current_status += f"* **Current batch:** {len(batch_files)} images\n"
                                    current_status += f"* **Progress:** {processed_count + error_count}/{len(image_files)} images\n"
                                    current_status += f"* **Completed:** {processed_count} SUCCESS\n"
                                    current_status += f"* **Errors:** {error_count} ERROR\n"
                                    current_status += f"* **Elapsed time:** {elapsed/60:.1f} minutes\n"
                                    current_status += f"* **Avg time/image:** {avg_time:.1f}s\n"
                                    if remaining_total > 0:
                                        current_status += f"* **ETA:** {eta/60:.1f} minutes\n"
                                    current_status += f"* **Workers active:** {len(remaining_futures)}\n\n"
                                    
                                    # Show currently processing files (not completed)
                                    current_status += f"Refresh **Currently Processing:**\n"
                                    active_files = [Path(future_to_path[f]).name for f in remaining_futures[:5]]
                                    if len(remaining_futures) > 5:
                                        active_files.append(f"... +{len(remaining_futures) - 5} more")
                                    for i, filename in enumerate(active_files, 1):
                                        if not filename.startswith("..."):
                                            current_status += f"   {i}. `{filename}`\n"
                                        else:
                                            current_status += f"   {filename}\n"
                                    
                                    yield current_status, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), self.get_stats()
                                    
                            except Exception as e:
                                error_count += 1
                else:
                    # Single-threaded processing
                    for img_path in batch_files:
                        if not self.processing_active:
                            break
                        result = process_single_image(str(img_path))
                        batch_results.append(result)
                        
                        if result["error"] is None:
                            processed_count += 1
                        else:
                            error_count += 1
                
                # Store results
                self.ocr._store_batch_results(batch_results)
            
            # Final status
            total_time = time.time() - start_time
            final_status = f"SUCCESS **Processing Complete!**\n\n"
            final_status += f"* **Total time:** {total_time/60:.1f} minutes\n"
            final_status += f"* **Successfully processed:** {processed_count}\n"
            final_status += f"* **Errors:** {error_count}\n"
            final_status += f"* **Average time per image:** {total_time/(processed_count + error_count):.1f}s\n"
            final_status += f"* **Workers used:** {num_workers}\n"
            final_status += f"* **Throughput:** {(processed_count + error_count)/(total_time/60):.1f} images/min"
            
            yield final_status, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), self.get_stats()
                   
        except Exception as e:
            yield f"ERROR **Error during processing:** {str(e)}", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), self.get_stats()
        finally:
            self.processing_active = False
    
    def process_new_images(self, num_workers: int = 10, progress=gr.Progress()):
        """Process only new images that haven't been processed yet."""
        if not OCR_AVAILABLE:
            return "ERROR OCR libraries not installed"
        
        # Get unprocessed images
        unprocessed = self.ocr.get_unprocessed_images()
        
        if not unprocessed:
            yield "SUCCESS No new images found to process!", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), self.get_stats()
            return
        
        self.processing_active = True
        processed_count = 0
        error_count = 0
        start_time = time.time()
        
        try:
            # Initialize status
            yield f"NEW **Processing New Images**\n\n* Found {len(unprocessed)} new images\n* Using {num_workers} workers\n* Starting parallel processing...", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), self.get_stats()
            
            # Process in batches
            batch_size = 50
            total_batches = (len(unprocessed) - 1) // batch_size + 1
            
            for batch_idx in range(total_batches):
                if not self.processing_active:
                    yield f"STOP **Processing Stopped**\n\n* Processed: {processed_count}\n* Errors: {error_count}\n* Status: Stopped by user", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), self.get_stats()
                    return
                
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(unprocessed))
                batch_files = unprocessed[start_idx:end_idx]
                
                # Update status
                elapsed = time.time() - start_time
                avg_time = elapsed / max(processed_count + error_count, 1)
                remaining = len(unprocessed) - (processed_count + error_count)
                eta = avg_time * remaining
                
                status = f"Processing **Processing New Images - Batch {batch_idx + 1}/{total_batches}**\n\n"
                status += f"* **Current batch:** {len(batch_files)} images\n"
                status += f"* **Progress:** {processed_count + error_count}/{len(unprocessed)} images\n"
                status += f"* **Completed:** {processed_count} SUCCESS\n"
                status += f"* **Errors:** {error_count} ERROR\n"
                status += f"* **Elapsed time:** {elapsed/60:.1f} minutes\n"
                status += f"* **Avg time/image:** {avg_time:.1f}s\n"
                if remaining > 0:
                    status += f"* **ETA:** {eta/60:.1f} minutes\n"
                status += f"* **Workers active:** {num_workers}\n\n"
                
                # Process batch
                batch_results = []
                
                if num_workers > 1:
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        future_to_path = {executor.submit(process_single_image, str(path)): path 
                                        for path in batch_files}
                        
                        # Initial status with all submitted files
                        status += f"Refresh **Currently Processing:**\n"
                        all_files = [Path(str(f)).name for f in batch_files]
                        display_files = all_files[:5]
                        if len(all_files) > 5:
                            display_files.append(f"... +{len(all_files) - 5} more")
                        for i, filename in enumerate(display_files, 1):
                            if not filename.startswith("..."):
                                status += f"   {i}. `{filename}`\n"
                            else:
                                status += f"   {filename}\n"
                        
                        yield status, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), self.get_stats()
                        
                        for future in as_completed(future_to_path):
                            try:
                                result = future.result()
                                batch_results.append(result)
                                
                                if result["error"] is None:
                                    processed_count += 1
                                else:
                                    error_count += 1
                                
                                # Update status with remaining active processes
                                remaining_futures = [f for f in future_to_path.keys() if not f.done()]
                                if remaining_futures:
                                    elapsed = time.time() - start_time
                                    avg_time = elapsed / max(processed_count + error_count, 1)
                                    remaining_total = len(unprocessed) - (processed_count + error_count)
                                    eta = avg_time * remaining_total
                                    
                                    current_status = f"Processing **Processing New Images - Batch {batch_idx + 1}/{total_batches}**\n\n"
                                    current_status += f"* **Current batch:** {len(batch_files)} images\n"
                                    current_status += f"* **Progress:** {processed_count + error_count}/{len(unprocessed)} images\n"
                                    current_status += f"* **Completed:** {processed_count} SUCCESS\n"
                                    current_status += f"* **Errors:** {error_count} ERROR\n"
                                    current_status += f"* **Elapsed time:** {elapsed/60:.1f} minutes\n"
                                    current_status += f"* **Avg time/image:** {avg_time:.1f}s\n"
                                    if remaining_total > 0:
                                        current_status += f"* **ETA:** {eta/60:.1f} minutes\n"
                                    current_status += f"* **Workers active:** {len(remaining_futures)}\n\n"
                                    
                                    # Show currently processing files (not completed)
                                    current_status += f"Refresh **Currently Processing:**\n"
                                    active_files = [Path(future_to_path[f]).name for f in remaining_futures[:5]]
                                    if len(remaining_futures) > 5:
                                        active_files.append(f"... +{len(remaining_futures) - 5} more")
                                    for i, filename in enumerate(active_files, 1):
                                        if not filename.startswith("..."):
                                            current_status += f"   {i}. `{filename}`\n"
                                        else:
                                            current_status += f"   {filename}\n"
                                    
                                    yield current_status, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), self.get_stats()
                                    
                            except Exception as e:
                                error_count += 1
                else:
                    # Single-threaded
                    for img_path in batch_files:
                        if not self.processing_active:
                            break
                        result = process_single_image(str(img_path))
                        batch_results.append(result)
                        
                        if result["error"] is None:
                            processed_count += 1
                        else:
                            error_count += 1
                
                # Store results
                self.ocr._store_batch_results(batch_results)
            
            # Final status
            total_time = time.time() - start_time
            final_status = f"SUCCESS **New Images Processing Complete!**\n\n"
            final_status += f"* **Total time:** {total_time/60:.1f} minutes\n"
            final_status += f"* **Successfully processed:** {processed_count}\n"
            final_status += f"* **Errors:** {error_count}\n"
            final_status += f"* **Average time per image:** {total_time/(processed_count + error_count):.1f}s\n"
            final_status += f"* **Workers used:** {num_workers}\n"
            final_status += f"* **Throughput:** {(processed_count + error_count)/(total_time/60):.1f} images/min"
            
            yield final_status, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), self.get_stats()
                   
        except Exception as e:
            yield f"ERROR **Error during processing:** {str(e)}", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), self.get_stats()
        finally:
            self.processing_active = False
    
    def stop_processing(self):
        """Stop the current processing."""
        self.processing_active = False
        return "STOP Processing stopped by user", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)
    
    def find_duplicates(self) -> List[List[Dict]]:
        """Find duplicate memes based on file hash."""
        conn = sqlite3.connect(self.ocr.db_path)
        cursor = conn.cursor()
        
        # Find all file hashes that appear more than once
        cursor.execute("""
            SELECT file_hash, COUNT(*) as count
            FROM memes
            WHERE file_hash != ''
            GROUP BY file_hash
            HAVING count > 1
            ORDER BY count DESC
        """)
        
        duplicate_groups = []
        
        for file_hash, count in cursor.fetchall():
            # Get all memes with this hash
            cursor.execute("""
                SELECT filename, filepath, file_hash, cleaned_text, 
                       confidence_score, created_at, processed_at
                FROM memes
                WHERE file_hash = ?
                ORDER BY created_at ASC
            """, (file_hash,))
            
            duplicates = []
            for row in cursor.fetchall():
                duplicates.append({
                    'filename': row[0],
                    'filepath': row[1],
                    'file_hash': row[2],
                    'text': row[3] or '',
                    'confidence': row[4],
                    'created_at': row[5],
                    'processed_at': row[6],
                    'exists': os.path.exists(row[1])
                })
            
            if len(duplicates) > 1:
                duplicate_groups.append(duplicates)
        
        conn.close()
        return duplicate_groups
    
    def delete_duplicates(self, filepaths_to_delete: List[str]) -> Dict[str, Any]:
        """Delete selected duplicate memes."""
        deleted_count = 0
        errors = []
        
        for filepath in filepaths_to_delete:
            try:
                result = self.ocr.delete_meme(filepath)
                if result['success']:
                    deleted_count += 1
                else:
                    errors.append(f"{filepath}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                errors.append(f"{filepath}: {str(e)}")
        
        return {
            'success': len(errors) == 0,
            'deleted_count': deleted_count,
            'errors': errors,
            'message': f"Deleted {deleted_count} duplicate(s)" + (f" with {len(errors)} error(s)" if errors else "")
        }
    
    def handle_duplicate_deletion(self, paths_json: str):
        """Handle duplicate deletion from the UI."""
        try:
            import json
            paths = json.loads(paths_json)
            result = self.delete_duplicates(paths)
            
            # Refresh the duplicate display
            return self.display_duplicates()
        except Exception as e:
            return gr.update(), f"❌ Error deleting duplicates: {str(e)}"
    
    def display_duplicates(self):
        """Find and display duplicate memes."""
        duplicate_groups = self.find_duplicates()
        
        if not duplicate_groups:
            return gr.update(value=""), "✅ No duplicates found!"
        
        # Create HTML for duplicate display
        import base64
        
        html_content = f"""
        <div style="margin-top: 20px;">
            <h3>Found {len(duplicate_groups)} groups of duplicates ({sum(len(group) for group in duplicate_groups)} total files)</h3>
            <p style="color: #666; margin-bottom: 20px;">The first file in each group (with green border) is considered the original based on creation date.</p>
        """
        
        for group_idx, duplicates in enumerate(duplicate_groups):
            # Get list of deletable filepaths for this group
            deletable_paths = [dup['filepath'] for dup in duplicates[1:] if dup['exists']]
            
            html_content += f"""
            <div style="margin: 20px 0; padding: 20px; background: #f8f8f8; border-radius: 10px; border: 1px solid #e0e0e0;">
                <h4>Duplicate Group {group_idx + 1} - {len(duplicates)} files (Hash: {duplicates[0]['file_hash'][:8]}...)</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 15px;">
            """
            
            for idx, dup in enumerate(duplicates):
                # Read image and convert to base64 if file exists
                img_html = ""
                if dup['exists'] and os.path.exists(dup['filepath']):
                    try:
                        with open(dup['filepath'], 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        img_html = f'<img src="data:image/jpeg;base64,{img_data}" style="width: 100%; height: auto; max-height: 200px; object-fit: contain; border-radius: 8px;">'
                    except:
                        img_html = '<div style="background: #ddd; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px;">Image not available</div>'
                else:
                    img_html = '<div style="background: #ffcccc; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px;">File missing</div>'
                
                # Determine if this is the original (first created)
                is_original = idx == 0
                
                html_content += f"""
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); {'border: 2px solid #4CAF50;' if is_original else ''}" id="dup-{group_idx}-{idx}">
                    {img_html}
                    <div style="margin-top: 10px; font-size: 12px;">
                        <strong>{dup['filename']}</strong><br>
                        <span style="color: #666;">Created: {dup['created_at']}</span><br>
                        <span style="color: #666;">Confidence: {dup['confidence']:.2f}</span><br>
                        {'<span style="color: #4CAF50; font-weight: bold;">✓ Original (oldest)</span>' if is_original else '<span style="color: #f44336;">● Duplicate</span>'}
                        {'' if dup['exists'] else '<br><span style="color: #f44336;">⚠️ File missing</span>'}
                    </div>
                </div>
                """
            
            if deletable_paths:
                html_content += f"""
                    </div>
                    <div style="margin-top: 15px; padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
                        <p style="margin: 0 0 10px 0; color: #856404;">
                            <strong>⚠️ Action Required:</strong> {len(deletable_paths)} duplicate(s) can be deleted in this group.
                        </p>
                        <p style="margin: 0 0 10px 0; color: #666; font-size: 14px;">
                            Click the button below to delete all duplicates and keep only the original (oldest) file.
                        </p>
                        <button class="delete-group-btn" data-group="{group_idx}" data-paths='{json.dumps(deletable_paths)}' 
                                style="background: #dc3545; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: 500;">
                            🗑️ Delete {len(deletable_paths)} Duplicate(s)
                        </button>
                    </div>
                """
            else:
                html_content += """
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;">
                        ✓ No deletable duplicates in this group
                    </div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </div>
        """
        
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups)  # -1 for original in each group
        info_text = f"Found {len(duplicate_groups)} duplicate groups with {total_duplicates} duplicate files that can be deleted."
        
        return gr.update(value=html_content), info_text
    
    
    def upload_and_process(self, files, num_workers: int = 4, progress=gr.Progress()):
        """Process uploaded files."""
        if not files:
            return "ERROR No files uploaded"
        
        self.processing_active = True
        processed_count = 0
        error_count = 0
        
        try:
            progress(0, desc=f"Processing {len(files)} uploaded files...")
            
            # Process uploaded files
            batch_results = []
            
            for i, file in enumerate(files):
                if not self.processing_active:
                    break
                
                progress(i / len(files), desc=f"Processing {file.name}...")
                
                try:
                    result = process_single_image(file.name)
                    batch_results.append(result)
                    
                    if result["error"] is None:
                        processed_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
            
            # Store results
            if batch_results:
                self.ocr._store_batch_results(batch_results)
            
            return (f"SUCCESS **Upload Processing Complete!**\n\n"
                   f"* **Successfully processed:** {processed_count}\n"
                   f"* **Errors:** {error_count}")
                   
        except Exception as e:
            return f"ERROR Error during upload processing: {str(e)}"
        finally:
            self.processing_active = False


def create_interface():
    """Create the Gradio interface."""
    gui = MemeGUI()
    
    # JavaScript that will be injected in CSS to ensure it loads
    head_js = """
    <script>
    console.log('Loading menu functions...');
    
    // Define global functions immediately
    window.toggleMenu = function(index) {
        console.log('toggleMenu called with index:', index);
        var menu = document.getElementById('menu-' + index);
        var allMenus = document.querySelectorAll('[id^="menu-"]');
        
        // Close all other menus
        for (var i = 0; i < allMenus.length; i++) {
            if (allMenus[i].id !== 'menu-' + index) {
                allMenus[i].style.display = 'none';
            }
        }
        
        // Toggle current menu
        if (menu) {
            if (menu.style.display === 'none' || menu.style.display === '') {
                menu.style.display = 'block';
                console.log('Menu opened');
            } else {
                menu.style.display = 'none';
                console.log('Menu closed');
            }
        } else {
            console.error('Menu not found:', 'menu-' + index);
        }
    };
    
    window.copyImage = function(index) {
        console.log('copyImage called with index:', index);
        document.getElementById('menu-' + index).style.display = 'none';
        
        var img = document.querySelector('#menu-' + index).parentElement.querySelector('img');
        var menuBtn = document.getElementById('menu-btn-' + index);
        
        try {
            // Create a canvas to convert the image
            var canvas = document.createElement('canvas');
            var ctx = canvas.getContext('2d');
            var tempImg = new Image();
            tempImg.crossOrigin = 'anonymous';
            
            tempImg.onload = function() {
                try {
                    canvas.width = tempImg.naturalWidth;
                    canvas.height = tempImg.naturalHeight;
                    ctx.drawImage(tempImg, 0, 0);
                    
                    canvas.toBlob(function(blob) {
                        try {
                            var clipboardItem = new ClipboardItem({'image/png': blob});
                            navigator.clipboard.write([clipboardItem]).then(function() {
                                // Show success feedback
                                menuBtn.innerHTML = '✓';
                                menuBtn.style.background = 'rgba(76, 175, 80, 0.8)';
                                setTimeout(function() {
                                    menuBtn.innerHTML = '⋮';
                                    menuBtn.style.background = 'rgba(0, 0, 0, 0.75)';
                                }, 1500);
                                
                                // Show success toast
                                var toast = document.createElement('div');
                                toast.style.cssText = 'position: fixed; top: 20px; right: 20px; background: white; border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 16px; z-index: 100000; font-family: system-ui, sans-serif; transform: translateX(100%); transition: transform 0.3s ease;';
                                toast.innerHTML = '<div style="display: flex; align-items: center; gap: 12px;"><div style="width: 24px; height: 24px; border-radius: 50%; background: #4CAF50; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">✓</div><span style="color: #333; font-weight: 500;">Image copied to clipboard!</span></div>';
                                document.body.appendChild(toast);
                                setTimeout(function() { toast.style.transform = 'translateX(0)'; }, 10);
                                setTimeout(function() {
                                    toast.style.transform = 'translateX(100%)';
                                    setTimeout(function() { toast.remove(); }, 300);
                                }, 3000);
                            }).catch(function() {
                                throw new Error('Clipboard API failed');
                            });
                        } catch(e) {
                            throw new Error('Failed to create clipboard item');
                        }
                    }, 'image/png');
                } catch(e) {
                    throw new Error('Failed to process image');
                }
            };
            
            tempImg.onerror = function() {
                throw new Error('Failed to load image');
            };
            
            tempImg.src = img.getAttribute('data-original');
            
        } catch(e) {
            // Show error feedback
            menuBtn.innerHTML = '✕';
            menuBtn.style.background = 'rgba(244, 67, 54, 0.8)';
            setTimeout(function() {
                menuBtn.innerHTML = '⋮';
                menuBtn.style.background = 'rgba(0, 0, 0, 0.75)';
            }, 1500);
            
            // Show error toast
            var toast = document.createElement('div');
            toast.style.cssText = 'position: fixed; top: 20px; right: 20px; background: white; border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 16px; z-index: 100000; font-family: system-ui, sans-serif; transform: translateX(100%); transition: transform 0.3s ease;';
            toast.innerHTML = '<div style="display: flex; align-items: center; gap: 12px;"><div style="width: 24px; height: 24px; border-radius: 50%; background: #f44336; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">✕</div><span style="color: #333; font-weight: 500;">Failed to copy image</span></div>';
            document.body.appendChild(toast);
            setTimeout(function() { toast.style.transform = 'translateX(0)'; }, 10);
            setTimeout(function() {
                toast.style.transform = 'translateX(100%)';
                setTimeout(function() { toast.remove(); }, 300);
            }, 3000);
        }
    };
    
    window.deleteImage = function(index, filepath) {
        console.log('deleteImage called with index:', index, 'filepath:', filepath);
        document.getElementById('menu-' + index).style.display = 'none';
        
        // Modern confirmation dialog
        var overlay = document.createElement('div');
        overlay.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.5); display: flex; align-items: center; justify-content: center; z-index: 200000; opacity: 0; transition: opacity 0.2s ease;';
        
        var dialog = document.createElement('div');
        dialog.style.cssText = 'background: white; border-radius: 12px; box-shadow: 0 20px 40px rgba(0,0,0,0.3); padding: 24px; min-width: 320px; max-width: 400px; transform: scale(0.9); transition: transform 0.2s ease; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;';
        
        dialog.innerHTML = '<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px;"><div style="width: 40px; height: 40px; border-radius: 50%; background: #ff5722; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px; font-weight: bold;">⚠</div><div><h3 style="margin: 0; font-size: 18px; font-weight: 600; color: #333;">Confirm Delete</h3><p style="margin: 4px 0 0 0; color: #666; font-size: 14px;">This will permanently delete the meme from the database and storage.</p></div></div><div style="display: flex; gap: 12px; justify-content: flex-end;"><button id="cancel-btn-' + index + '" style="background: #f5f5f5; border: none; border-radius: 6px; padding: 10px 20px; font-size: 14px; font-weight: 500; color: #333; cursor: pointer; transition: background-color 0.2s;">Cancel</button><button id="confirm-btn-' + index + '" style="background: #f44336; border: none; border-radius: 6px; padding: 10px 20px; font-size: 14px; font-weight: 500; color: white; cursor: pointer; transition: background-color 0.2s;">Delete</button></div>';
        
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        
        // Animate in
        setTimeout(function() {
            overlay.style.opacity = '1';
            dialog.style.transform = 'scale(1)';
        }, 10);
        
        var closeDialog = function() {
            overlay.style.opacity = '0';
            dialog.style.transform = 'scale(0.9)';
            setTimeout(function() {
                if (overlay.parentNode) {
                    overlay.remove();
                }
            }, 200);
        };
        
        // Handle button clicks
        document.getElementById('cancel-btn-' + index).onclick = closeDialog;
        
        document.getElementById('confirm-btn-' + index).onclick = function() {
            closeDialog();
            
            // User confirmed delete
            var imageDiv = document.querySelector('#menu-' + index).parentElement;
            imageDiv.style.opacity = '0.3';
            imageDiv.style.pointerEvents = 'none';
            
            // Show info toast
            var toast = document.createElement('div');
            toast.style.cssText = 'position: fixed; top: 20px; right: 20px; background: white; border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 16px; z-index: 100000; font-family: system-ui, sans-serif; transform: translateX(100%); transition: transform 0.3s ease; min-width: 300px;';
            toast.innerHTML = '<div style="display: flex; align-items: center; gap: 12px;"><div style="width: 24px; height: 24px; border-radius: 50%; background: #2196F3; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">ℹ</div><span style="color: #333; font-weight: 500;">Delete functionality requires backend implementation</span><button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: #666; cursor: pointer; font-size: 18px; margin-left: auto;">×</button></div>';
            document.body.appendChild(toast);
            setTimeout(function() { toast.style.transform = 'translateX(0)'; }, 10);
            setTimeout(function() {
                toast.style.transform = 'translateX(100%)';
                setTimeout(function() { toast.remove(); }, 300);
            }, 5000);
        };
        
        // Close on overlay click
        overlay.onclick = function(e) {
            if (e.target === overlay) {
                closeDialog();
            }
        };
        
        // Close on Escape key
        var handleEscape = function(e) {
            if (e.key === 'Escape') {
                closeDialog();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
    };
    
    // Close menus when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.menu-btn') && !event.target.closest('[id^="menu-"]')) {
            var allMenus = document.querySelectorAll('[id^="menu-"]');
            for (var i = 0; i < allMenus.length; i++) {
                allMenus[i].style.display = 'none';
            }
        }
    });
    
    console.log('Menu functions loaded successfully');
    </script>
    """
    
    # Custom CSS for modern, professional styling
    css = head_js + """
    /* Container and layout - full width */
    .gradio-container, .gradio-app, .app, .main, .container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 5px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Target Gradio's internal container classes */
    .gradio-container > .main, 
    .gradio-container .wrap,
    .gradio-container .container,
    div[data-testid="container"] {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 20px !important;
    }
    
    /* Root container override */
    .app > .main {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
    }
    
    /* Universal width override for any container-like elements */
    [class*="container"], [class*="wrap"], [class*="main"] {
        max-width: 100% !important;
    }
    
    /* Specific targeting for modern Gradio versions */
    .svelte-1gfkn6j, .svelte-container, .block-container {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Header will use inline styles */
    
    /* Stats box styling - multiple columns, no gradient */
    .stats-box {
        column-count: 3 !important;
        column-gap: 30px !important;
        column-fill: balance !important;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stats-box p {
        margin: 0 !important;
        break-inside: avoid !important;
    }
    
    /* Input containers - prevent wrapping */
    .gr-textbox, .gr-slider {
        min-width: 200px !important;
        flex-shrink: 0 !important;
    }
    
    /* Row containers - no wrap */
    .gr-row {
        flex-wrap: nowrap !important;
        gap: 15px !important;
        align-items: flex-end !important;
    }
    
    /* Search button styling */
    .gr-button {
        white-space: nowrap !important;
        min-width: 120px !important;
        flex-shrink: 0 !important;
    }
    
    /* Column sizing for search */
    .gr-column {
        min-width: 0 !important;
        flex-grow: 1 !important;
    }
    
    /* Tab styling */
    .tab-nav {
        background: white;
        border-radius: 12px;
        padding: 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .btn-secondary {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }
    
    .btn-stop {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }
    
    /* Input styling */
    .gr-textbox input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 12px 16px;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    
    .gr-textbox input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Gallery styling */
    .gallery {
        overflow-y: auto;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        background: white;
        padding: 15px;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Ensure HTML components don't overflow */
    .gr-html {
        width: 100% !important;
        overflow: hidden !important;
    }
    
    /* Image grid specific styling */
    #image-grid {
        max-width: 100% !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    #image-grid > div {
        min-width: 0 !important;
        overflow: hidden !important;
    }
    
    /* Gallery image container */
    #gallery img {
        object-fit: contain;
        width: 100%;
        height: auto;
        max-width: 100%;
        display: block;
    }
    
    /* Gallery grid items */
    #gallery .gallery-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    /* Ensure SVG icons are visible */
    #image-grid button svg {
        width: 20px !important;
        height: 20px !important;
        display: block !important;
    }
    
    /* Spinning animation for loading */
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Accordion styling */
    .gr-accordion {
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin-bottom: 15px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Status text styling */
    .status-text {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #667eea;
        font-family: 'Courier New', Courier, monospace;
        line-height: 1.6;
    }
    
    /* Refresh button centering */
    .refresh-button-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 120px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 10px !important;
        }
        
        .header-section {
            padding: 15px;
        }
        
        .stats-box {
            padding: 12px;
        }
    }
    """
    
    # JavaScript to inject at interface level
    js_code = """
    function() {
        console.log('Injecting menu functions via Gradio js...');
        
        // Define global functions
        window.toggleMenu = function(index, event) {
            console.log('toggleMenu called with index:', index);
            
            // Stop event propagation
            if (event) {
                event.stopPropagation();
            }
            
            // Close all menus first
            var allMenus = document.querySelectorAll('.context-menu-dynamic');
            for (var i = 0; i < allMenus.length; i++) {
                allMenus[i].remove();
            }
            
            // Check if menu should be opened or closed
            var existingMenu = document.querySelector('.context-menu-dynamic[data-index="' + index + '"]');
            if (existingMenu) {
                existingMenu.remove();
                console.log('Menu closed');
                return false;
            }
            
            // Create new menu element
            var menuBtn = document.getElementById('menu-btn-' + index);
            if (!menuBtn) {
                console.error('Menu button not found');
                return false;
            }
            
            var container = menuBtn.parentElement;
            var img = container.querySelector('img');
            var imgPath = img ? img.getAttribute('data-filepath') : '';
            
            // Create menu HTML
            var menu = document.createElement('div');
            menu.className = 'context-menu-dynamic';
            menu.setAttribute('data-index', index);
            menu.style.cssText = 'position: absolute; top: 42px; left: 8px; background: #ffffff; border: 2px solid #e0e0e0; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); padding: 8px; min-width: 160px; z-index: 99999;';
            
            menu.innerHTML = `
                <div style="padding: 12px 16px; cursor: pointer; display: flex; align-items: center; gap: 12px; border-radius: 6px; transition: background-color 0.2s; color: #333; font-size: 14px; font-weight: 500;"
                     onmouseover="this.style.backgroundColor='#f5f5f5'" 
                     onmouseout="this.style.backgroundColor='transparent'"
                     onclick="window.copyImage(${index})">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: #2196F3;">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                    <span>Copy Image</span>
                </div>
                <div style="padding: 12px 16px; cursor: pointer; display: flex; align-items: center; gap: 12px; border-radius: 6px; transition: background-color 0.2s; color: #d32f2f; font-size: 14px; font-weight: 500;"
                     onmouseover="this.style.backgroundColor='#ffebee'" 
                     onmouseout="this.style.backgroundColor='transparent'"
                     onclick="window.deleteImage(${index}, '${imgPath}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: #f44336;">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                    <span>Delete</span>
                </div>
            `;
            
            container.appendChild(menu);
            console.log('Menu created and opened');
            
            return false;
        };
        
        window.copyImage = function(index) {
            console.log('copyImage called with index:', index);
            
            try {
                // Remove the dynamic menu
                var menus = document.querySelectorAll('.context-menu-dynamic');
                for (var i = 0; i < menus.length; i++) {
                    menus[i].remove();
                }
                console.log('Menu removed');
                
                // Find the image and button directly by their IDs/relationships
                var menuBtn = document.getElementById('menu-btn-' + index);
                var img = menuBtn ? menuBtn.parentElement.querySelector('img') : null;
                
                console.log('Found img:', img);
                console.log('Found menuBtn:', menuBtn);
                
                if (!menuBtn) {
                    console.error('Menu button not found!');
                    return;
                }
                
                // Try to copy image to clipboard
                console.log('Attempting to copy image to clipboard...');
                
                // Create a canvas to convert the image
                var canvas = document.createElement('canvas');
                var ctx = canvas.getContext('2d');
                var tempImg = new Image();
                tempImg.crossOrigin = 'anonymous';
                
                tempImg.onload = function() {
                    try {
                        canvas.width = tempImg.naturalWidth;
                        canvas.height = tempImg.naturalHeight;
                        ctx.drawImage(tempImg, 0, 0);
                        
                        canvas.toBlob(function(blob) {
                            if (blob && typeof ClipboardItem !== 'undefined') {
                                // Try modern clipboard API
                                var clipboardItem = new ClipboardItem({'image/png': blob});
                                navigator.clipboard.write([clipboardItem]).then(function() {
                                    console.log('Image copied to clipboard successfully!');
                                    // Show success feedback
                                    menuBtn.innerHTML = '✓';
                                    menuBtn.style.background = 'rgba(76, 175, 80, 0.8)';
                                    setTimeout(function() {
                                        menuBtn.innerHTML = '⋮';
                                        menuBtn.style.background = 'rgba(0, 0, 0, 0.75)';
                                    }, 1500);
                                    
                                    // Show success toast
                                    var toast = document.createElement('div');
                                    toast.style.cssText = 'position: fixed; top: 20px; right: 20px; background: white; border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 16px; z-index: 100000; font-family: system-ui, sans-serif; transform: translateX(100%); transition: transform 0.3s ease;';
                                    toast.innerHTML = '<div style="display: flex; align-items: center; gap: 12px;"><div style="width: 24px; height: 24px; border-radius: 50%; background: #4CAF50; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">✓</div><span style="color: #333; font-weight: 500;">Image copied to clipboard!</span></div>';
                                    document.body.appendChild(toast);
                                    setTimeout(function() { toast.style.transform = 'translateX(0)'; }, 10);
                                    setTimeout(function() {
                                        toast.style.transform = 'translateX(100%)';
                                        setTimeout(function() { toast.remove(); }, 300);
                                    }, 3000);
                                }).catch(function(err) {
                                    console.error('Clipboard API failed:', err);
                                    throw err;
                                });
                            } else {
                                throw new Error('Clipboard API not available');
                            }
                        }, 'image/png');
                    } catch(e) {
                        console.error('Canvas error:', e);
                        throw e;
                    }
                };
                
                tempImg.onerror = function() {
                    console.error('Failed to load image');
                    throw new Error('Failed to load image');
                };
                
                tempImg.src = img.getAttribute('data-original');
                console.log('Image loading initiated...');
            } catch (error) {
                console.error('Error in copyImage:', error);
            }
        };
        
        window.deleteImage = function(index, filepath) {
            console.log('deleteImage called with index:', index, 'filepath:', filepath);
            
            // Remove the dynamic menu
            var menus = document.querySelectorAll('.context-menu-dynamic');
            for (var i = 0; i < menus.length; i++) {
                menus[i].remove();
            }
            
            // Show confirmation dialog
            var confirmDelete = confirm('Are you sure you want to delete this meme?\\n\\nThis will remove it from the database and disk permanently.');
            
            if (confirmDelete) {
                // Find the Gradio textbox and update its value
                var textboxes = document.querySelectorAll('textarea[data-testid="textbox"]');
                var deleteTextbox = null;
                
                // Find the hidden delete filepath textbox
                for (var i = 0; i < textboxes.length; i++) {
                    var parent = textboxes[i].closest('.hidden');
                    if (parent || textboxes[i].style.display === 'none') {
                        deleteTextbox = textboxes[i];
                        break;
                    }
                }
                
                if (deleteTextbox) {
                    // Update the textbox value
                    deleteTextbox.value = filepath;
                    deleteTextbox.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // Trigger the delete button click
                    setTimeout(function() {
                        var deleteBtn = document.querySelector('button#hidden-delete-btn');
                        if (deleteBtn) {
                            deleteBtn.click();
                        }
                    }, 100);
                } else {
                    console.error('Delete textbox not found');
                }
                
                // Disable the image while deleting
                var imageDiv = document.querySelector('#menu-btn-' + index).parentElement;
                imageDiv.style.opacity = '0.3';
                imageDiv.style.pointerEvents = 'none';
            }
        };
        
        // Close menus when clicking outside
        document.addEventListener('click', function(event) {
            if (!event.target.closest('.menu-btn') && !event.target.closest('.context-menu-dynamic')) {
                var allMenus = document.querySelectorAll('.context-menu-dynamic');
                for (var i = 0; i < allMenus.length; i++) {
                    allMenus[i].remove();
                }
            }
        });
        
        // Add duplicate deletion handler
        document.addEventListener('click', function(event) {
            if (event.target.classList.contains('delete-group-btn')) {
                var paths = event.target.getAttribute('data-paths');
                var pathsArray = JSON.parse(paths);
                
                if (confirm('Delete ' + pathsArray.length + ' duplicate file(s)? This action cannot be undone.')) {
                    // Update the hidden textbox
                    var pathsTextbox = document.querySelector('#duplicate-paths textarea');
                    if (pathsTextbox) {
                        pathsTextbox.value = paths;
                        pathsTextbox.dispatchEvent(new Event('input', { bubbles: true }));
                        
                        // Trigger the delete button
                        setTimeout(function() {
                            var deleteBtn = document.querySelector('#trigger-duplicate-delete');
                            if (deleteBtn) {
                                deleteBtn.click();
                            }
                        }, 100);
                    }
                }
            }
        });
        
        console.log('Menu functions injected successfully via Gradio js');
    }
    """

    with gr.Blocks(
        title="Meme OCR Search System", 
        theme=gr.themes.Base(),
        css=css,
        js=js_code
    ) as interface:
        
        # Header with visual separation
        with gr.Row():
            gr.HTML("""
            <div style="
                width: 100%;
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                text-align: center;
            ">
                <h1 style="margin: 0; font-size: 2rem;">🔍 Search Meme OCR Search System</h1>
                <p style="margin: 8px 0 0 0; opacity: 0.7;">Extract text from memes and search through your collection with AI-powered OCR</p>
            </div>
            """)
        
        # Statistics with visual separation
        stats_display = gr.HTML(
            value=gui.get_stats()
        )
        
        # Main tabs
        with gr.Tabs():
            
            # Search Tab
            with gr.TabItem("🔍 Search Memes", id="search"):
                gr.Markdown("**Search Query**")
                with gr.Row():
                    with gr.Column(scale=6, min_width=300):
                        search_query = gr.Textbox(
                            label="",
                            placeholder="Enter text to search for in memes...",
                            container=False
                        )
                    with gr.Column(scale=1, min_width=120):
                        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        max_results = gr.Slider(
                            minimum=5, maximum=200, value=100, step=5,
                            label="Max Results"
                        )
                    with gr.Column():
                        min_confidence = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                            label="Min Confidence"
                        )
                    with gr.Column():
                        preview_size = gr.Slider(
                            minimum=100, maximum=500, value=200, step=50,
                            label="Preview Size (px)"
                        )
                
                # Single HTML grid that does everything
                search_results_html = gr.HTML(
                    value="",
                    visible=True,
                    label="Search Results"
                )
                
                search_info = gr.Markdown(
                    value="Enter a search term above to find memes.",
                    label="Results Info"
                )
                
                # Hidden components for delete functionality
                with gr.Row(visible=False):
                    delete_filepath = gr.Textbox(elem_id="delete-filepath", visible=False)
                    delete_result = gr.JSON(visible=False)
                    hidden_delete_btn = gr.Button("Delete", elem_id="hidden-delete-btn", visible=False)
            
            # Processing Tab
            with gr.TabItem("⚙️ Process Images", id="process"):
                
                with gr.Accordion("📁 Process Images", open=True):
                    with gr.Row():
                        with gr.Column(scale=3):
                            folder_path = gr.Textbox(
                                label="Folder Path",
                                value=gui.default_memes_path,
                                placeholder=gui.default_memes_path
                            )
                        with gr.Column(scale=1):
                            folder_workers = gr.Slider(
                                minimum=1, maximum=16, value=10, step=1,
                                label="Workers"
                            )
                    
                    with gr.Row():
                        process_new_btn = gr.Button("Process New Only", variant="primary", elem_classes=["btn-primary"])
                        process_folder_btn = gr.Button("Folder Process All", variant="secondary", elem_classes=["btn-secondary"])
                        stop_btn = gr.Button("STOP Stop", variant="stop", elem_classes=["btn-stop"], interactive=False)
                    
                    folder_status = gr.Markdown(
                        value="Click 'NEW Process New Only' to quickly process only new images, or 'Folder Process All' to reprocess everything.",
                        label="Processing Status",
                        elem_classes=["status-text"]
                    )
                
                with gr.Accordion("Upload Upload Files", open=False):
                    file_upload = gr.File(
                        label="Upload Image Files",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    gr.Markdown("Upload individual image files to process")
                    
                    with gr.Row():
                        upload_workers = gr.Slider(
                            minimum=1, maximum=8, value=2, step=1,
                            label="Workers"
                        )
                        upload_btn = gr.Button("Upload Process Uploads", variant="primary")
                    
                    upload_status = gr.Markdown(
                        value="Upload image files above to process them.",
                        label="Upload Status"
                    )
            
            # Maintenance Tab
            with gr.TabItem("🛠️ Maintenance", id="maintenance"):
                with gr.Accordion("🔍 Find Duplicate Memes", open=True):
                    gr.Markdown("""
                    This tool finds duplicate memes based on their file hash. 
                    Duplicates are files with identical content, even if they have different names.
                    """)
                    
                    find_duplicates_btn = gr.Button("🔍 Find Duplicates", variant="primary")
                    
                    duplicates_html = gr.HTML(
                        value="",
                        label="Duplicate Groups",
                        elem_id="duplicates-display"
                    )
                    
                    duplicates_info = gr.Markdown(
                        value="Click 'Find Duplicates' to scan for duplicate memes.",
                        label="Status"
                    )
                    
                    # Hidden components for duplicate deletion
                    with gr.Row(visible=False):
                        duplicate_paths_to_delete = gr.Textbox(visible=False, elem_id="duplicate-paths")
                        trigger_duplicate_delete = gr.Button("Trigger Delete", visible=False, elem_id="trigger-duplicate-delete")
            
            # System Tab
            with gr.TabItem("⚙️ System", id="system"):
                gr.HTML("""
                <div style="padding: 20px;">
                    <h3>System Settings</h3>
                    <p>Current configuration and system information.</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"""
                        ### System Info
                        - **OCR Available:** {"✅ Yes" if OCR_AVAILABLE else "❌ No"}
                        - **CPU Cores:** {mp.cpu_count()}
                        - **Recommended Workers:** {min(mp.cpu_count(), 8)}
                        
                        ### Database
                        - **Location:** `meme_index.db`
                        - **Type:** SQLite with FTS5
                        
                        ### Supported Formats
                        - JPG, JPEG, PNG, GIF
                        - BMP, TIFF, WebP
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### Tips for Better Results
                        
                        **OCR Performance:**
                        - Higher resolution images work better
                        - Clear, readable text improves accuracy
                        - Avoid heavily stylized fonts
                        
                        **Search Tips:**
                        - Use simple, common words
                        - Try partial words if exact matches fail
                        - Lower confidence threshold for more results
                        
                        **Processing:**
                        - More workers = faster processing
                        - Don't exceed your CPU core count
                        - Processing new images only is fastest
                        """)
        
        # Event handlers
        search_btn.click(
            fn=gui.search_memes,
            inputs=[search_query, max_results, min_confidence, preview_size],
            outputs=[search_results_html, search_info]
        )
        
        search_query.submit(
            fn=gui.search_memes,
            inputs=[search_query, max_results, min_confidence, preview_size],
            outputs=[search_results_html, search_info]
        )
        
        # Processing event handlers with button state management and stats refresh
        process_folder_btn.click(
            fn=gui.process_folder,
            inputs=[folder_path, folder_workers],
            outputs=[folder_status, process_folder_btn, process_new_btn, stop_btn, stats_display]
        )
        
        process_new_btn.click(
            fn=gui.process_new_images,
            inputs=[folder_workers],
            outputs=[folder_status, process_folder_btn, process_new_btn, stop_btn, stats_display]
        )
        
        upload_btn.click(
            fn=gui.upload_and_process,
            inputs=[file_upload, upload_workers],
            outputs=[upload_status]
        )
        
        stop_btn.click(
            fn=gui.stop_processing,
            outputs=[folder_status, process_folder_btn, process_new_btn, stop_btn]
        )
        
        # Maintenance tab handlers
        find_duplicates_btn.click(
            fn=gui.display_duplicates,
            outputs=[duplicates_html, duplicates_info]
        )
        
        trigger_duplicate_delete.click(
            fn=gui.handle_duplicate_deletion,
            inputs=[duplicate_paths_to_delete],
            outputs=[duplicates_html, duplicates_info]
        )
        
        # Delete handler with automatic refresh
        def handle_delete_and_refresh(filepath, current_query, max_results, min_confidence, preview_size):
            """Handle delete and refresh search results."""
            result = gui.delete_meme(filepath)
            
            # Show toast notification via JavaScript
            if result["success"]:
                toast_js = f"""
                var toast = document.createElement('div');
                toast.style.cssText = 'position: fixed; top: 20px; right: 20px; background: white; border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 16px; z-index: 100000; font-family: system-ui, sans-serif; transform: translateX(100%); transition: transform 0.3s ease;';
                toast.innerHTML = '<div style="display: flex; align-items: center; gap: 12px;"><div style="width: 24px; height: 24px; border-radius: 50%; background: #4CAF50; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">✓</div><span style="color: #333; font-weight: 500;">{result["message"]}</span></div>';
                document.body.appendChild(toast);
                setTimeout(function() {{ toast.style.transform = 'translateX(0)'; }}, 10);
                setTimeout(function() {{
                    toast.style.transform = 'translateX(100%)';
                    setTimeout(function() {{ toast.remove(); }}, 300);
                }}, 3000);
                """
            else:
                toast_js = f"""
                var toast = document.createElement('div');
                toast.style.cssText = 'position: fixed; top: 20px; right: 20px; background: white; border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 16px; z-index: 100000; font-family: system-ui, sans-serif; transform: translateX(100%); transition: transform 0.3s ease;';
                toast.innerHTML = '<div style="display: flex; align-items: center; gap: 12px;"><div style="width: 24px; height: 24px; border-radius: 50%; background: #f44336; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">✕</div><span style="color: #333; font-weight: 500;">Error: {result.get("error", "Unknown error")}</span></div>';
                document.body.appendChild(toast);
                setTimeout(function() {{ toast.style.transform = 'translateX(0)'; }}, 10);
                setTimeout(function() {{
                    toast.style.transform = 'translateX(100%)';
                    setTimeout(function() {{ toast.remove(); }}, 300);
                }}, 3000);
                """
            
            # Refresh search results if we have a query
            if current_query:
                search_html, search_msg = gui.search_memes(current_query, max_results, min_confidence, preview_size)
                # Inject the toast JavaScript into the HTML
                if isinstance(search_html, dict) and 'value' in search_html:
                    search_html['value'] = f"<script>{toast_js}</script>" + search_html['value']
                else:
                    # If it's a gr.update object, get the value
                    html_value = search_html.value if hasattr(search_html, 'value') else str(search_html)
                    search_html = gr.update(value=f"<script>{toast_js}</script>" + html_value)
                return result, search_html, search_msg
            else:
                # Just return empty results with toast
                return result, gr.update(value=f"<script>{toast_js}</script>"), "Enter a search term above to find memes."
        
        hidden_delete_btn.click(
            fn=handle_delete_and_refresh,
            inputs=[delete_filepath, search_query, max_results, min_confidence, preview_size],
            outputs=[delete_result, search_results_html, search_info]
        )
        
        # Load initial stats
        interface.load(
            fn=gui.get_stats,
            outputs=[stats_display]
        )
    
    return interface


def main():
    """Launch the Gradio interface."""
    if not OCR_AVAILABLE:
        print("⚠️  OCR libraries not available. Please install them first:")
        print("pip install pytesseract Pillow opencv-python")
        print("Also install Tesseract OCR system package.")
        return
    
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    print("Starting Starting Meme OCR Web Interface...")
    
    interface = create_interface()
    
    # Get the memes directory path for allowed_paths
    memes_path = Path(__file__).parent.absolute() / "memes"
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public URL
        show_error=True,
        favicon_path=None,
        allowed_paths=[str(memes_path)],  # Allow access to memes directory
        app_kwargs={"title": "Meme OCR Search System"}
    )


if __name__ == "__main__":
    main()