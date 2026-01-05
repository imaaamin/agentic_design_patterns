"""
Arabic Document Parsing with PaddleOCR-VL

A comprehensive solution for parsing Arabic official documents (PDFs/images)
using PaddleOCR's Vision-Language model and traditional OCR capabilities.

PaddleOCR-VL: Ultra-compact 0.9B Vision-Language Model for multilingual document parsing
- Supports 100+ languages including Arabic
- Can extract structured data from complex documents
- Outputs to JSON and Markdown formats

Usage:
    # Basic usage - parse a PDF with VL model (best for understanding)
    uv run parsing_docs_with_paddleocr.py document.pdf

    # Use traditional OCR (faster, text extraction only)
    uv run parsing_docs_with_paddleocr.py document.pdf --mode ocr

    # Save results to JSON
    uv run parsing_docs_with_paddleocr.py document.pdf --output results.json

    # Parse an image
    uv run parsing_docs_with_paddleocr.py document.png

References:
    - https://github.com/PaddlePaddle/PaddleOCR
    - PaddleOCR-VL: 0.9B Ultra-Compact Vision-Language Model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

# Set environment variable to skip model source check (speeds up startup)
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")


@dataclass
class TextLine:
    """A single line of detected text."""
    text: str
    confidence: float
    bbox: list


@dataclass
class PageResult:
    """Results for a single page."""
    page_number: int
    lines: list[TextLine] = field(default_factory=list)
    text: str = ""
    raw_content: Optional[dict] = None


@dataclass
class ParseResult:
    """Container for document parsing results."""
    source_file: str
    parse_mode: str
    timestamp: str
    pages: list[PageResult]
    raw_text: str
    metadata: dict


def ensure_paddleocr_installed() -> bool:
    """Check if PaddleOCR is installed."""
    try:
        import paddleocr
        return True
    except ImportError as e:
        print(f"❌ PaddleOCR import error: {e}")
        print("\n📦 To install PaddleOCR, run:")
        print("   uv pip install paddleocr paddlepaddle")
        return False


def convert_pdf_to_images(pdf_path: str, dpi: int = 150) -> list[tuple[str, int]]:
    """Convert PDF pages to images using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        print("❌ PyMuPDF required for PDF processing.")
        print("   Install with: uv add pymupdf")
        return []
    
    doc = fitz.open(pdf_path)
    images = []
    
    print(f"   📚 Converting {len(doc)} pages to images (DPI: {dpi})...")
    
    temp_dir = "/tmp/paddleocr_pages"
    os.makedirs(temp_dir, exist_ok=True)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        
        base_name = Path(pdf_path).stem
        temp_img_path = f"{temp_dir}/{base_name}_page_{page_num + 1}.png"
        pix.save(temp_img_path)
        
        images.append((temp_img_path, page_num + 1))
        print(f"      Page {page_num + 1}: {pix.width}x{pix.height} px")
    
    doc.close()
    return images


def parse_with_paddleocr_vl(
    file_path: str,
    output_dir: Optional[str] = None
) -> ParseResult:
    """
    Parse document using PaddleOCR-VL (Vision-Language model).
    
    PaddleOCR-VL is a 0.9B parameter vision-language model optimized for
    multilingual document understanding and structured data extraction.
    """
    from paddleocr import PaddleOCRVL
    
    print(f"\n🤖 Initializing PaddleOCR-VL...")
    print(f"   Model: PaddleOCR Vision-Language (0.9B)")
    
    pipeline = PaddleOCRVL()
    
    print(f"\n📄 Processing: {file_path}")
    output = pipeline.predict(file_path)
    
    pages = []
    all_text = []
    
    for idx, res in enumerate(output):
        print(f"\n   📖 Page {idx + 1}:")
        
        page_data = PageResult(
            page_number=idx + 1,
            lines=[],
            text="",
            raw_content=None
        )
        
        try:
            # Save to files if output_dir specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                res.save_to_json(save_path=output_dir)
                res.save_to_markdown(save_path=output_dir)
                print(f"   💾 Saved to: {output_dir}")
            
            # Extract content from result - VL returns dict-like results
            res_dict = None
            if hasattr(res, 'json') and res.json:
                res_dict = res.json
            elif hasattr(res, '__getitem__'):
                # Result might be dict-like
                try:
                    res_dict = dict(res)
                except:
                    pass
            
            if res_dict:
                page_data.raw_content = res_dict
                # Extract text from parsing_res_list
                parsing_data = res_dict.get('res', res_dict) if isinstance(res_dict, dict) else res_dict
                if isinstance(parsing_data, dict):
                    parsing_list = parsing_data.get('parsing_res_list', [])
                    page_texts = []
                    lines = []
                    for block in parsing_list:
                        if isinstance(block, dict):
                            content = block.get('block_content', '')
                            label = block.get('block_label', '')
                            bbox = block.get('block_bbox', [])
                            if content and content.strip():
                                page_texts.append(content)
                                lines.append(LineResult(
                                    text=content,
                                    confidence=1.0,
                                    bbox=bbox,
                                    label=label
                                ))
                    page_data.lines = lines
                    page_data.text = "\n".join(page_texts)
                    if page_texts:
                        all_text.append(page_data.text)
                        print(f"      ✓ Extracted {len(page_texts)} text blocks")
            
            # Fallback: try text/markdown attributes
            if not page_data.text:
                if hasattr(res, 'text') and res.text and isinstance(res.text, str):
                    page_data.text = res.text
                    all_text.append(res.text)
                elif hasattr(res, 'markdown') and res.markdown and isinstance(res.markdown, str):
                    page_data.text = res.markdown
                    all_text.append(res.markdown)
                
        except Exception as e:
            print(f"   ⚠️ Error processing page {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
        
        pages.append(page_data)
    
    result = ParseResult(
        source_file=file_path,
        parse_mode="paddleocr_vl",
        timestamp=datetime.now().isoformat(),
        pages=pages,
        raw_text="\n\n".join(all_text),
        metadata={
            "model": "PaddleOCR-VL",
            "num_pages": len(pages)
        }
    )
    
    print(f"\n✅ VL parsing complete: {len(pages)} page(s) processed")
    return result


def parse_with_paddleocr(
    file_path: str,
    lang: str = "ar",
    use_textline_orientation: bool = True,
    device: str = "cpu",
    dpi: int = 150
) -> ParseResult:
    """
    Parse document using traditional PaddleOCR (PP-OCRv5).
    
    PP-OCRv5 provides high-accuracy OCR with support for 80+ languages
    including Arabic with RTL text support.
    """
    from paddleocr import PaddleOCR
    
    print(f"\n🔤 Initializing PaddleOCR (PP-OCRv5)...")
    print(f"   Language: {lang}")
    print(f"   Textline Orientation: {use_textline_orientation}")
    print(f"   Device: {device}")
    
    ocr = PaddleOCR(
        use_textline_orientation=use_textline_orientation,
        lang=lang,
        device=device,
    )
    
    print(f"\n📄 Processing: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        image_files = convert_pdf_to_images(file_path, dpi=dpi)
        if not image_files:
            return ParseResult(
                source_file=file_path,
                parse_mode="paddleocr",
                timestamp=datetime.now().isoformat(),
                pages=[],
                raw_text="",
                metadata={"error": "Failed to convert PDF to images"}
            )
    else:
        image_files = [(file_path, 1)]
    
    pages = []
    all_text = []
    
    for img_path, page_num in image_files:
        print(f"\n   📖 Processing page {page_num}...")
        
        # Use the new predict API
        result = ocr.predict(img_path)
        
        lines = []
        page_text_parts = []
        
        # Handle the OCRResult format (dict-like object)
        try:
            for res in result:
                # Access dict-like keys
                rec_texts = res.get('rec_texts', []) if hasattr(res, 'get') else []
                rec_scores = res.get('rec_scores', []) if hasattr(res, 'get') else []
                dt_polys = res.get('dt_polys', []) if hasattr(res, 'get') else []
                
                for i, text in enumerate(rec_texts):
                    confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                    bbox = dt_polys[i].tolist() if i < len(dt_polys) else []
                    
                    lines.append(TextLine(
                        text=text,
                        confidence=confidence,
                        bbox=bbox
                    ))
                    page_text_parts.append(text)
        except Exception as e:
            print(f"      ⚠️ Error parsing result: {e}")
            import traceback
            traceback.print_exc()
        
        if lines:
            print(f"      ✓ Found {len(lines)} text lines")
            preview = " | ".join(page_text_parts[:3])
            if len(preview) > 80:
                preview = preview[:80] + "..."
            print(f"      Preview: {preview}")
        else:
            print(f"      ⚠️ No text detected")
        
        page_text = "\n".join(page_text_parts)
        
        pages.append(PageResult(
            page_number=page_num,
            lines=lines,
            text=page_text
        ))
        all_text.append(f"--- Page {page_num} ---\n{page_text}")
    
    # Clean up temp images
    if file_ext == '.pdf':
        for img_path, _ in image_files:
            try:
                os.remove(img_path)
            except:
                pass
    
    total_lines = sum(len(p.lines) for p in pages)
    avg_confidence = 0
    if total_lines > 0:
        avg_confidence = sum(
            line.confidence for p in pages for line in p.lines
        ) / total_lines
    
    result = ParseResult(
        source_file=file_path,
        parse_mode="paddleocr",
        timestamp=datetime.now().isoformat(),
        pages=pages,
        raw_text="\n\n".join(all_text),
        metadata={
            "model": "PP-OCRv5",
            "language": lang,
            "num_pages": len(pages),
            "total_lines": total_lines,
            "avg_confidence": round(avg_confidence, 3)
        }
    )
    
    print(f"\n✅ OCR complete!")
    print(f"   Pages: {len(pages)}, Lines: {total_lines}, Confidence: {avg_confidence:.1%}")
    
    return result


def parse_with_pp_structure(
    file_path: str,
    output_dir: Optional[str] = None
) -> ParseResult:
    """
    Parse document using PP-StructureV3 for layout analysis.
    
    PP-StructureV3 provides:
    - Layout detection (text, titles, figures, tables)
    - Table structure recognition
    - Document recovery to markdown
    """
    from paddleocr import PPStructureV3
    
    print(f"\n📊 Initializing PP-StructureV3...")
    
    structure_engine = PPStructureV3(
        lang='ar',
    )
    
    print(f"\n📄 Processing: {file_path}")
    result = structure_engine.predict(file_path)
    
    pages = []
    all_text = []
    total_tables = 0
    
    for idx, page_result in enumerate(result):
        print(f"\n   📖 Page {idx + 1}:")
        
        page_text_parts = []
        tables = []
        raw_content = {}
        
        # Handle new API format
        try:
            # Get the result dict
            res_dict = page_result
            if hasattr(page_result, 'get'):
                res_dict = page_result
            elif hasattr(page_result, 'json'):
                res_dict = page_result.json
            
            # Extract text from parsing_res_list (structured blocks)
            parsing_res = res_dict.get('parsing_res_list', [])
            for block in parsing_res:
                # Handle both dict and object formats
                if hasattr(block, 'get'):
                    block_label = block.get('block_label', '')
                    block_content = block.get('block_content', '')
                    block_bbox = block.get('block_bbox', [])
                else:
                    # Object with attributes
                    block_label = getattr(block, 'block_label', getattr(block, 'label', ''))
                    block_content = getattr(block, 'block_content', getattr(block, 'content', ''))
                    block_bbox = getattr(block, 'block_bbox', getattr(block, 'bbox', []))
                
                if block_label == 'table':
                    tables.append({
                        "bbox": block_bbox,
                        "html": block_content,
                    })
                    total_tables += 1
                elif block_content:
                    # Add text content
                    if block_label in ['doc_title', 'title', 'paragraph_title']:
                        page_text_parts.append(f"# {block_content}")
                    else:
                        page_text_parts.append(str(block_content))
            
            # Also get OCR texts for more complete extraction
            ocr_res = res_dict.get('overall_ocr_res', {})
            rec_texts = ocr_res.get('rec_texts', [])
            if rec_texts and not page_text_parts:
                # Use OCR texts if no structured blocks
                page_text_parts = [t for t in rec_texts if t.strip()]
            
            # Store raw content for reference
            raw_content = {
                'width': res_dict.get('width'),
                'height': res_dict.get('height'),
                'page_index': res_dict.get('page_index'),
                'num_blocks': len(parsing_res),
                'rec_texts': rec_texts,
            }
            
            print(f"      ✓ Found {len(parsing_res)} blocks, {len(rec_texts)} text lines")
                
        except Exception as e:
            print(f"      ⚠️ Error parsing structure result: {e}")
            import traceback
            traceback.print_exc()
        
        page_text = "\n".join(page_text_parts)
        all_text.append(page_text)
        
        pages.append(PageResult(
            page_number=idx + 1,
            text=page_text,
            raw_content={"tables": tables, "num_regions": len(raw_content), "raw": raw_content}
        ))
        
        print(f"      Found {len(tables)} tables")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "structure_result.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"pages": [p.raw_content for p in pages]}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Saved to: {output_path}")
    
    return ParseResult(
        source_file=file_path,
        parse_mode="pp_structure",
        timestamp=datetime.now().isoformat(),
        pages=pages,
        raw_text="\n\n".join(all_text),
        metadata={
            "model": "PP-StructureV3",
            "num_pages": len(pages),
            "total_tables": total_tables
        }
    )


def save_results(result: ParseResult, output_path: str, format: str = "json"):
    """Save parsing results to file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if format == "json":
        data = {
            "source_file": result.source_file,
            "parse_mode": result.parse_mode,
            "timestamp": result.timestamp,
            "metadata": result.metadata,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "lines": [
                        {"text": l.text, "confidence": l.confidence, "bbox": l.bbox}
                        for l in p.lines
                    ] if p.lines else [],
                    "raw_content": p.raw_content
                }
                for p in result.pages
            ],
            "raw_text": result.raw_text
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    elif format == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.raw_text)
            
    elif format == "md":
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Document Parse Results\n\n")
            f.write(f"**Source:** `{result.source_file}`\n")
            f.write(f"**Mode:** {result.parse_mode}\n")
            f.write(f"**Timestamp:** {result.timestamp}\n\n")
            f.write(f"---\n\n")
            for page in result.pages:
                f.write(f"## Page {page.page_number}\n\n")
                f.write(page.text)
                f.write("\n\n")
    
    print(f"\n💾 Results saved to: {output_path}")


def print_results(result: ParseResult):
    """Print a summary of parsing results."""
    print("\n" + "=" * 60)
    print("📊 PARSING RESULTS")
    print("=" * 60)
    
    print(f"\n📁 Source: {result.source_file}")
    print(f"🔧 Mode: {result.parse_mode}")
    print(f"📅 Timestamp: {result.timestamp}")
    
    print(f"\n📋 Metadata:")
    for key, value in result.metadata.items():
        if isinstance(value, float) and 'confidence' in key:
            print(f"   {key}: {value:.1%}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n📄 Pages: {len(result.pages)}")
    
    if result.raw_text:
        preview_len = min(500, len(result.raw_text))
        print(f"\n📝 Text Preview ({len(result.raw_text)} total chars):")
        print("-" * 40)
        print(result.raw_text[:preview_len])
        if len(result.raw_text) > preview_len:
            print("...")
        print("-" * 40)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Arabic documents using PaddleOCR-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse with Vision-Language model (best for understanding)
  uv run parsing_docs_with_paddleocr.py document.pdf
  
  # Parse with traditional OCR (faster)
  uv run parsing_docs_with_paddleocr.py document.pdf --mode ocr
  
  # Parse for layout/table analysis
  uv run parsing_docs_with_paddleocr.py document.pdf --mode structure
  
  # Save results
  uv run parsing_docs_with_paddleocr.py document.pdf --output results.json
        """
    )
    
    parser.add_argument("file_path", help="Path to PDF or image file")
    parser.add_argument(
        "--mode", "-m",
        choices=["vl", "ocr", "structure"],
        default="vl",
        help="Parsing mode: vl (Vision-Language), ocr (PP-OCRv5), structure (layout analysis)"
    )
    parser.add_argument("--lang", "-l", default="ar", help="Language for OCR mode (default: ar)")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "txt", "md"], default="json")
    parser.add_argument("--output-dir", "-d", help="Directory for PaddleOCR native outputs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Device to use (default: cpu)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF rendering")
    
    args = parser.parse_args()
    
    if not ensure_paddleocr_installed():
        sys.exit(1)
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File not found: {args.file_path}")
        sys.exit(1)
    
    try:
        print("=" * 60)
        print("🔍 PaddleOCR Arabic Document Parser")
        print("=" * 60)
        
        if args.mode == "vl":
            result = parse_with_paddleocr_vl(args.file_path, args.output_dir)
        elif args.mode == "ocr":
            result = parse_with_paddleocr(
                args.file_path, args.lang, device=args.device, dpi=args.dpi
            )
        elif args.mode == "structure":
            result = parse_with_pp_structure(args.file_path, args.output_dir)
        
        print_results(result)
        
        if args.output:
            output_ext = Path(args.output).suffix.lower()
            if output_ext == ".md":
                args.format = "md"
            elif output_ext == ".txt":
                args.format = "txt"
            save_results(result, args.output, args.format)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
