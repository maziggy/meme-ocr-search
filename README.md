# Meme OCR Search System

A powerful Python-based system for extracting text from meme images using OCR (Optical Character Recognition) and creating a searchable database with an intuitive web interface.

## Preface

This project addresses a common problem for meme enthusiasts: finding specific memes in large collections. By extracting text from meme images and creating a searchable database, you can quickly locate any meme by searching for words or phrases that appear in it. The system features both command-line tools and a modern web interface built with Gradio.

## Features

- **🔍 Web Interface**: Modern, user-friendly GUI for searching and managing memes
- **📝 Text Extraction**: Uses Tesseract OCR to extract text from meme images
- **🖼️ Image Preprocessing**: Applies denoising and thresholding for better OCR accuracy
- **⚡ Full-Text Search**: SQLite FTS (Full-Text Search) for lightning-fast text queries
- **🔄 Incremental Processing**: Only processes new images, not already-processed ones
- **🗂️ Duplicate Detection**: MD5 hashing to identify and manage duplicate images
- **⚙️ Multiple OCR Methods**: Tries different OCR configurations for best results
- **📊 Confidence Scoring**: Rates text quality to filter out poor extractions
- **🖼️ Multiple Formats**: Supports JPG, PNG, GIF, BMP, TIFF, and WebP images
- **🛠️ Maintenance Tools**: Built-in duplicate finder and management
- **📋 Context Menus**: Right-click to copy or delete memes
- **📱 Responsive Design**: Works on desktop and mobile devices

## Installation

### Prerequisites

1. **Install Tesseract OCR**:

   **macOS**:
   ```bash
   brew install tesseract
   ```

   **Linux (Ubuntu/Debian)**:
   ```bash
   sudo apt-get install tesseract-ocr
   ```

   **Windows**:
   Download from: https://github.com/UB-Mannheim/tesseract/wiki

2. **Python 3.7+** is required

### Setup

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd meme-db
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   python setup.py
   ```

4. **Place your meme images** in the `memes/` directory

## Running

### Web Interface (Recommended)

Launch the web interface for the best experience:

```bash
python meme_gui.py
```

This will start a local web server (usually at `http://localhost:7860`) with a modern interface featuring:
- **Search Tab**: Find memes with advanced filtering options
- **Process Images Tab**: Add new memes and process them
- **Maintenance Tab**: Find and manage duplicate memes
- **System Tab**: View system information and statistics

### Command Line Interface

For advanced users or automation:

```bash
# Extract text from all memes (first run)
python meme_ocr.py --extract

# Process only new images (recommended for regular use)
python meme_ocr.py --update

# Search for specific text
python meme_ocr.py --search "funny text"

# View database statistics
python meme_ocr.py --stats
```

## Using

### First Time Setup

1. **Add your memes** to the `memes/` folder
2. **Launch the web interface**: `python meme_gui.py`
3. **Go to "Process Images" tab**
4. **Click "Folder Process All"** to process all your memes initially
5. **Wait for processing to complete** (you'll see progress updates)

### Daily Usage

#### Searching for Memes

1. **Go to the "Search Memes" tab**
2. **Enter search terms** in the search box (e.g., "drake pointing", "this is fine")
3. **Adjust settings**:
   - **Max Results**: How many memes to show (default: 100)
   - **Min Confidence**: Filter out low-quality text extraction (0.1-1.0)
   - **Preview Size**: Thumbnail size (100-800px)
4. **Click "Search"** or press Enter
5. **Right-click any meme** to copy to clipboard or delete

#### Adding New Memes

**Method 1: Folder Processing**
1. Add new meme files to the `memes/` folder
2. Go to "Process Images" tab
3. Click "Process New Only" to process just the new files

**Method 2: File Upload**
1. Go to "Process Images" tab
2. Use the file upload area to drag/drop or select files
3. Click "Upload Process Uploads"

#### Managing Duplicates

1. **Go to "Maintenance" tab**
2. **Click "Find Duplicates"**
3. **Review duplicate groups** - the oldest file is marked as "Original"
4. **Click "Delete X Duplicate(s)"** to remove duplicates and keep originals

### Advanced Features

#### Search Tips

- **Simple search**: `drake meme`
- **Partial words**: `surpris` (finds "surprised")
- **Multiple terms**: `pikachu face` (finds memes with both words)
- **Exact phrases**: Use quotes if supported by your search terms

#### Processing Options

- **Workers**: Increase for faster processing (don't exceed CPU cores)
- **Force reprocess**: Check "Process All" to reprocess existing memes
- **Custom directory**: Place memes in any folder and specify the path

#### Context Menu Features

Right-click on any meme thumbnail to:
- **Copy Image**: Copies the meme to your clipboard
- **Delete**: Permanently removes the meme from database and disk

### Web Interface Tabs

#### 🔍 Search Memes
- Search your meme collection
- Filter by confidence and results count
- Adjust thumbnail size
- Context menu actions (copy/delete)

#### 🖼️ Process Images
- Process entire folder or new images only
- Upload individual files
- Monitor processing progress
- Adjust worker count for performance

#### 🛠️ Maintenance
- Find and manage duplicate memes
- View duplicate groups side-by-side
- Bulk delete duplicates while preserving originals

#### ⚙️ System
- View system information and OCR status
- Check database statistics
- Performance recommendations

## Command Line Reference

```bash
# Process all images in memes folder
python meme_ocr.py --extract

# Process only new/unprocessed images
python meme_ocr.py --update

# Search for text in memes
python meme_ocr.py --search "search terms"

# Force reprocess all images
python meme_ocr.py --extract --force

# Use custom memes directory
python meme_ocr.py --update --memes-dir "/path/to/memes"

# Show database statistics
python meme_ocr.py --stats

# Show help
python meme_ocr.py --help
```

## Troubleshooting

### "Tesseract not found" error
- Ensure Tesseract OCR is installed and in your system PATH
- On Windows, you may need to set the `TESSDATA_PREFIX` environment variable

### Poor OCR accuracy
- Use high-quality, high-resolution images
- Ensure text is clearly visible and not heavily stylized
- Check that images aren't corrupted or too compressed

### Web interface won't load
- Make sure port 7860 isn't being used by another application
- Check the console output for any error messages
- Try accessing `http://127.0.0.1:7860` instead of `localhost`

### No search results
- Verify text extraction worked by checking the "System" tab statistics
- Try broader or different search terms
- Check the confidence threshold isn't set too high

### Performance issues
- Reduce the number of workers if system becomes unresponsive
- Process images in smaller batches
- Consider upgrading system memory for large collections

## Technical Details

### How It Works

1. **Image Preprocessing**: Images are converted to grayscale, denoised, and thresholded
2. **OCR Extraction**: Tesseract OCR extracts text using optimized configurations
3. **Database Storage**: Text is stored in SQLite with full-text search indexing
4. **Web Interface**: Gradio provides the modern web interface
5. **Search**: Fast full-text search with confidence-based filtering

### Database Schema

- `memes`: Stores image metadata, file paths, extracted text, and processing info
- `memes_fts`: Full-text search virtual table for fast text queries

### Dependencies

- **pytesseract**: Python wrapper for Tesseract OCR
- **Pillow (PIL)**: Image processing library
- **opencv-python**: Computer vision library for preprocessing
- **gradio**: Web interface framework
- **sqlite3**: Database (included with Python)

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## License

This project is open source. Please check the license file for details.