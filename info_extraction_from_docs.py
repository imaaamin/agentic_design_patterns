"""
Arabic Document Information Extraction Workflow (Dynamic Schema)

Parses Arabic documents (Saudi Ministry of Commerce registrations) using Docling,
then uses a 4-agent pipeline:
1. Classifier Agent - Identifies the document type
2. Schema Agent - Dynamically identifies which fields to extract based on content
3. Extractor Agent - Extracts the fields identified by the Schema Agent
4. Verifier Agent - Reviews extracted numbers against document images for accuracy

Architecture:
    PDF → Docling → Classifier → Schema → Extractor → Verifier → Verified Data
                       ↓            ↓          ↓           ↓
                   doc_type    field_list   raw_data   corrections

Usage:
    uv run info_extraction_from_docs.py document.pdf
    uv run info_extraction_from_docs.py document.pdf --output results.json
    uv run info_extraction_from_docs.py document.pdf --multimodal
"""

import asyncio
import json
import os
import sys
from typing import TypedDict, Annotated, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# --- API Key Setup ---
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env")
    sys.exit(1)
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = api_key


# --- Pydantic Models ---
class FieldDefinition(BaseModel):
    """Definition of a field to extract."""
    name: str = Field(description="Field name in snake_case (e.g., company_name)")
    description: str = Field(description="What this field represents")
    arabic_label: Optional[str] = Field(None, description="Arabic label if found in document")
    data_type: str = Field(default="string", description="Expected data type: string, number, date, etc.")
    required: bool = Field(default=False, description="Whether this field is likely present")


class DynamicSchema(BaseModel):
    """Dynamically generated schema for extraction."""
    document_type: str
    fields: list[FieldDefinition]
    extraction_notes: str = Field(description="Any special notes for extraction")


class ExtractionResult(BaseModel):
    """Complete extraction result with final corrected data."""
    document_type: str
    confidence: float
    detected_language: str
    schema_fields: list[dict]
    extracted_data: dict  # Original extraction
    verified_data: Optional[dict] = None  # After verification
    verification_report: Optional[list[dict]] = None  # Corrections made
    final_data: dict = None  # THE CORRECTED OUTPUT (verified if available, else extracted)
    raw_text_preview: str


# --- Workflow State ---
class ExtractionState(TypedDict):
    """State for the extraction workflow."""
    document_path: str
    document_content: str
    page_images: Optional[list]
    use_multimodal: bool
    document_type: Optional[str]
    classification_confidence: Optional[float]
    detected_language: Optional[str]
    classification_reasoning: Optional[str]
    schema_fields: Optional[list]
    extraction_notes: Optional[str]
    extracted_data: Optional[dict]
    verified_data: Optional[dict]
    verification_report: Optional[list]
    messages: Annotated[list, add_messages]


# --- Document Parser ---
def parse_document_with_docling(file_path: str, verbose: bool = True) -> str:
    print(f"📄 Parsing document: {file_path}")
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
        
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            force_full_page_ocr=True,
        )
        
        pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: pdf_format_option}
        )
        
        result = converter.convert(file_path)
        document = result.document
        num_pages = len(document.pages) if hasattr(document, 'pages') and document.pages else 0
        content = document.export_to_markdown()
        
        print(f"✅ Parsed document:")
        print(f"   Pages: {num_pages}")
        print(f"   Total characters: {len(content)}")
        
        if verbose and num_pages > 0:
            print(f"\n   📖 Page-by-page breakdown:")
            lines = content.split('\n')
            print(f"   Total lines: {len(lines)}")
            print(f"\n   📄 Start of document:")
            print(f"   {content[:300]}...")
            if len(content) > 1000:
                mid_point = len(content) // 2
                print(f"\n   📄 Middle of document (char {mid_point}):")
                print(f"   ...{content[mid_point:mid_point+300]}...")
            if len(content) > 600:
                print(f"\n   📄 End of document:")
                print(f"   ...{content[-300:]}")
        
        return content
    except ImportError as e:
        print(f"⚠️ Docling import error: {e}")
        return parse_pdf_fallback(file_path)


def extract_pdf_pages_as_images(file_path: str) -> list[tuple[bytes, int]]:
    print(f"🖼️  Extracting pages as images: {file_path}")
    try:
        import fitz
        doc = fitz.open(file_path)
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            images.append((img_bytes, page_num + 1))
            print(f"   Page {page_num + 1}: {len(img_bytes)} bytes")
        doc.close()
        print(f"✅ Extracted {len(images)} page images")
        return images
    except ImportError:
        print("❌ PyMuPDF not installed")
        return []
    except Exception as e:
        print(f"❌ Error extracting images: {e}")
        return []


def parse_pdf_fallback(file_path: str) -> str:
    print(f"📄 Parsing with PyMuPDF: {file_path}")
    try:
        import fitz
        doc = fitz.open(file_path)
        num_pages = len(doc)
        print(f"✅ Opened document: {num_pages} pages")
        
        all_text = []
        total_chars = 0
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_content = f"\n{'='*40}\n📄 PAGE {page_num + 1}\n{'='*40}\n"
            for block in blocks:
                if block["type"] == 0:
                    for line in block.get("lines", []):
                        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                        if line_text.strip():
                            page_content += line_text + "\n"
                elif block["type"] == 1:
                    page_content += "[IMAGE]\n"
            
            tables = page.find_tables()
            if tables.tables:
                page_content += f"\n[{len(tables.tables)} table(s) found]\n"
                for i, table in enumerate(tables.tables):
                    page_content += f"\n--- Table {i+1} ---\n"
                    try:
                        df = table.to_pandas()
                        page_content += df.to_string() + "\n"
                    except:
                        for row in table.extract():
                            page_content += " | ".join([str(c) if c else "" for c in row]) + "\n"
            
            all_text.append(page_content)
            total_chars += len(page_content)
            print(f"   Page {page_num + 1}: {len(page_content)} characters")
        
        doc.close()
        content = "\n".join(all_text)
        print(f"\n✅ Total extracted: {total_chars} characters from {num_pages} pages")
        print(f"\n   📄 Start of document:")
        print(f"   {content[:400]}...")
        return content
    except ImportError:
        print("❌ PyMuPDF not installed. Run: uv add PyMuPDF")
        return f"[Could not extract text from {file_path}]"
    except Exception as e:
        print(f"❌ PyMuPDF error: {e}")
        return f"[Error extracting text: {e}]"


# --- LLM Setup ---
def get_llm(temperature: float = 0.1, model: str = "gemini-2.0-flash"):
    """Get LLM with specified model."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


# Default models for each agent
CLASSIFIER_MODEL = "gemini-2.0-flash"
SCHEMA_MODEL = "gemini-2.0-flash"
EXTRACTOR_MODEL = "gemini-3-pro-preview"
VERIFIER_MODEL = "gemini-3-pro-preview"  # Different model for verification


import base64
import re as regex_module


# --- Helper: Parse JSON from model response ---
def extract_text_from_response(response_content) -> str:
    """
    Extract text from model response content.
    
    Handles both:
    - Plain string response
    - List of content parts (multimodal format): [{"type": "text", "text": "..."}]
    """
    if isinstance(response_content, str):
        return response_content
    
    if isinstance(response_content, list):
        # Multimodal response format - extract all text parts
        text_parts = []
        for part in response_content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    
    # Fallback - convert to string
    return str(response_content)


def parse_json_from_response(response_content, fallback: dict = None) -> dict:
    """
    Parse JSON from model response, handling markdown code blocks and multimodal format.
    """
    text = extract_text_from_response(response_content)
    
    # Remove markdown code blocks
    text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        # Try to find JSON object
        json_match = regex_module.search(r'\{.*\}', text, regex_module.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"   ⚠️ JSON parse error: {e}")
        return fallback if fallback else {"_parse_error": str(e), "raw_text": text[:500]}


def build_multimodal_message(text: str, images: list[tuple[bytes, int]], max_images: int = 5) -> HumanMessage:
    content = [{"type": "text", "text": text}]
    for img_bytes, page_num in images[:max_images]:
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        content.append({"type": "text", "text": f"\n[Page {page_num}]:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
    return HumanMessage(content=content)


# --- Agent 1: Classifier ---
CLASSIFIER_PROMPT = """You are an expert document classifier specializing in Arabic business documents.

Analyze the document and identify:
1. The type of document (e.g., commercial registration, tax certificate, license, contract, etc.)
2. The primary language (Arabic, English, or mixed)
3. Your confidence level (0-1)

Respond with a JSON object:
{
    "document_type": "descriptive type name",
    "confidence": 0.95,
    "detected_language": "ar/en/mixed",
    "reasoning": "brief explanation"
}"""


async def classifier_node(state: ExtractionState) -> dict:
    print("\n" + "="*50)
    print("🏷️  Agent 1: Classifier")
    print(f"   Model: {CLASSIFIER_MODEL}")
    llm = get_llm(model=CLASSIFIER_MODEL)
    
    if state.get('use_multimodal') and state.get('page_images'):
        print("   📷 Using multimodal analysis (images)")
        message = build_multimodal_message(
            f"Classify this document.\n\nExtracted text:\n{state['document_content'][:5000]}",
            state['page_images'], max_images=3
        )
        messages = [SystemMessage(content=CLASSIFIER_PROMPT), message]
    else:
        messages = [SystemMessage(content=CLASSIFIER_PROMPT),
                    HumanMessage(content=f"Classify this document:\n\n{state['document_content'][:30000]}")]
    
    response = await llm.ainvoke(messages)
    
    try:
        import re
        text = response.content.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        result = json.loads(json_match.group()) if json_match else json.loads(text)
    except:
        result = {"document_type": "unknown", "confidence": 0.5, "detected_language": "unknown", "reasoning": response.content}
    
    print(f"   Type: {result.get('document_type')}")
    print(f"   Confidence: {result.get('confidence', 0):.0%}")
    print(f"   Language: {result.get('detected_language')}")
    
    return {
        "document_type": result.get("document_type"),
        "classification_confidence": result.get("confidence"),
        "detected_language": result.get("detected_language"),
        "classification_reasoning": result.get("reasoning"),
        "messages": [AIMessage(content=f"Classified as: {result.get('document_type')}")]
    }


# --- Agent 2: Schema Generator ---
SCHEMA_PROMPT = """You are an expert at analyzing documents and identifying extractable information.

Identify ALL fields that can be extracted. For each field:
1. Give it a snake_case name
2. Describe what it contains
3. Note the Arabic label if visible
4. Specify data type (string, number, date, currency)
5. Mark if required

Respond with JSON:
{
    "fields": [{"name": "...", "description": "...", "arabic_label": "...", "data_type": "...", "required": true/false}],
    "extraction_notes": "..."
}"""


async def schema_node(state: ExtractionState) -> dict:
    print("\n" + "="*50)
    print("📐 Agent 2: Schema Generator")
    print(f"   Model: {SCHEMA_MODEL}")
    llm = get_llm(temperature=0.2, model=SCHEMA_MODEL)
    
    prompt_text = f"""Document Type: {state['document_type']}
Language: {state['detected_language']}

Extracted text:
{state['document_content'][:10000]}

Identify ALL extractable fields from ALL pages."""
    
    if state.get('use_multimodal') and state.get('page_images'):
        print("   📷 Using multimodal analysis (images)")
        message = build_multimodal_message(prompt_text, state['page_images'], max_images=5)
        messages = [SystemMessage(content=SCHEMA_PROMPT), message]
    else:
        messages = [SystemMessage(content=SCHEMA_PROMPT), HumanMessage(content=prompt_text)]
    
    response = await llm.ainvoke(messages)
    
    try:
        import re
        text = response.content.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        result = json.loads(json_match.group()) if json_match else json.loads(text)
    except Exception as e:
        print(f"   ⚠️ Parse error: {e}")
        result = {"fields": [{"name": "content", "description": "Raw content", "data_type": "string", "required": True}], "extraction_notes": ""}
    
    fields = result.get("fields", [])
    print(f"   Identified {len(fields)} fields:")
    for f in fields[:10]:
        arabic = f" ({f.get('arabic_label')})" if f.get('arabic_label') else ""
        print(f"      • {f.get('name')}{arabic}: {f.get('description', '')[:50]}")
    if len(fields) > 10:
        print(f"      ... and {len(fields) - 10} more fields")
    
    return {"schema_fields": fields, "extraction_notes": result.get("extraction_notes", ""),
            "messages": [AIMessage(content=f"Identified {len(fields)} fields")]}


# --- Agent 3: Extractor ---
EXTRACTOR_PROMPT = """You are an expert at extracting structured information from documents.

Extract the value for each field:
- Be precise - only extract what is clearly stated
- For dates, preserve the original format (Hijri or Gregorian)
- For numbers, include any relevant units
- If not found, use null
- Preserve Arabic text

Return a JSON object with field names as keys and extracted values as values."""


async def extractor_node(state: ExtractionState) -> dict:
    print("\n" + "="*50)
    print("📋 Agent 3: Extractor")
    print(f"   Model: {EXTRACTOR_MODEL}")
    llm = get_llm(temperature=0.0, model=EXTRACTOR_MODEL)
    
    field_descriptions = []
    for f in state['schema_fields']:
        desc = f"- {f['name']}: {f.get('description', 'No description')}"
        if f.get('arabic_label'):
            desc += f" (Arabic: {f['arabic_label']})"
        field_descriptions.append(desc)
    
    prompt_text = f"""Extracted text:
{state['document_content'][:10000]}

Fields to Extract:
{chr(10).join(field_descriptions)}

{state.get('extraction_notes', '')}

Extract values and return as JSON."""
    
    if state.get('use_multimodal') and state.get('page_images'):
        print("   📷 Using multimodal analysis (images)")
        message = build_multimodal_message(prompt_text, state['page_images'], max_images=5)
        messages = [SystemMessage(content=EXTRACTOR_PROMPT), message]
    else:
        messages = [SystemMessage(content=EXTRACTOR_PROMPT), HumanMessage(content=prompt_text)]
    
    response = await llm.ainvoke(messages)
    
    extracted = parse_json_from_response(response.content, fallback={"_error": "parse_failed"})
    if "_parse_error" in extracted or "_error" in extracted:
        print(f"   ⚠️ Could not parse extraction response")
        extracted = {}
    
    filled = sum(1 for v in extracted.values() if v is not None and v != "")
    print(f"   Extracted {filled}/{len(state['schema_fields'])} fields:")
    for key, value in extracted.items():
        if value is not None and value != "" and key != "raw_response":
            display = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
            print(f"      ✓ {key}: {display}")
    
    return {"extracted_data": extracted, "messages": [AIMessage(content=f"Extracted {filled} fields")]}


# --- Agent 4: Verifier ---
VERIFIER_PROMPT = """You are an expert data verification agent specializing in Arabic document analysis.

VERIFY the accuracy of extracted numerical data by comparing against the document images.

Focus on:
1. ID numbers (registration, national IDs, reference numbers)
2. Dates (Hijri and Gregorian formats)
3. Phone/fax numbers
4. Financial amounts
5. Postal codes
6. Any numerical sequences

For EACH field with numbers:
1. Find it in the document image
2. Read the actual value digit by digit
3. Compare with extracted value
4. Report any discrepancies

Watch for:
- ٥ (5) vs ٦ (6), ٧ (7) vs ٨ (8), ٠ (0) vs ٥ (5)
- Missing/extra digits
- Wrong date formats

Respond with JSON:
{
    "corrections": [
        {"field_name": "...", "original_value": "...", "corrected_value": "...", "confidence": 0.95, "reason": "..."}
    ],
    "verified_fields": ["list of correct fields"],
    "unverifiable_fields": ["fields that couldn't be verified"],
    "summary": "brief summary"
}"""


async def verifier_node(state: ExtractionState) -> dict:
    print("\n" + "="*50)
    print("🔍 Agent 4: Verifier")
    
    if not state.get('use_multimodal') or not state.get('page_images'):
        print("   ⚠️ Skipping verification (no images available)")
        return {"verified_data": state['extracted_data'], "verification_report": [],
                "messages": [AIMessage(content="Verification skipped - no images")]}
    
    print(f"   📷 Verifying with model: {VERIFIER_MODEL}")
    llm = get_llm(temperature=0.0, model=VERIFIER_MODEL)
    
    extracted = state['extracted_data']
    numerical_fields = []
    
    for field_name, value in extracted.items():
        if value is None or value == "":
            continue
        value_str = str(value)
        if any(c.isdigit() or c in '٠١٢٣٤٥٦٧٨٩' for c in value_str):
            numerical_fields.append({"field_name": field_name, "extracted_value": value_str})
    
    print(f"   Found {len(numerical_fields)} fields with numerical data to verify")
    
    if not numerical_fields:
        print("   ✓ No numerical fields to verify")
        return {"verified_data": extracted, "verification_report": [],
                "messages": [AIMessage(content="No numerical fields to verify")]}
    
    fields_to_verify = json.dumps(numerical_fields, ensure_ascii=False, indent=2)
    prompt_text = f"""Verify these extracted numerical values against the document images:

{fields_to_verify}

For each field:
1. Find where this data appears in the document
2. Read the actual value digit by digit
3. Compare with extracted value
4. Report any discrepancies

Be careful with Arabic numerals (٠١٢٣٤٥٦٧٨٩) vs Western (0123456789)."""
    
    message = build_multimodal_message(prompt_text, state['page_images'], max_images=5)
    messages = [SystemMessage(content=VERIFIER_PROMPT), message]
    response = await llm.ainvoke(messages)
    
    result = parse_json_from_response(
        response.content, 
        fallback={"corrections": [], "verified_fields": [], "unverifiable_fields": [], "summary": "Parse error"}
    )
    
    corrections = result.get("corrections", [])
    verified_data = extracted.copy()
    
    if corrections:
        print(f"\n   🔧 Found {len(corrections)} correction(s):")
        for corr in corrections:
            field = corr.get("field_name")
            original = corr.get("original_value")
            corrected = corr.get("corrected_value")
            reason = corr.get("reason", "")
            confidence = corr.get("confidence", 0)
            
            print(f"      ❌ {field}:")
            print(f"         Original:  {original}")
            print(f"         Corrected: {corrected}")
            print(f"         Reason: {reason} (confidence: {confidence:.0%})")
            
            if field in verified_data and corrected:
                verified_data[field] = corrected
    else:
        print("   ✅ All numerical fields verified correct!")
    
    verified_fields = result.get("verified_fields", [])
    if verified_fields:
        print(f"\n   ✓ Verified correct: {len(verified_fields)} fields")
        for vf in verified_fields[:5]:
            print(f"      • {vf}")
        if len(verified_fields) > 5:
            print(f"      ... and {len(verified_fields) - 5} more")
    
    unverifiable = result.get("unverifiable_fields", [])
    if unverifiable:
        print(f"\n   ⚠️ Could not verify: {unverifiable}")
    
    return {"verified_data": verified_data, "verification_report": corrections,
            "messages": [AIMessage(content=f"Verified {len(numerical_fields)} fields, {len(corrections)} corrections")]}


# --- Build Workflow ---
def build_extraction_workflow():
    graph = StateGraph(ExtractionState)
    graph.add_node("classifier", classifier_node)
    graph.add_node("schema_generator", schema_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("verifier", verifier_node)
    
    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "schema_generator")
    graph.add_edge("schema_generator", "extractor")
    graph.add_edge("extractor", "verifier")
    graph.add_edge("verifier", END)
    
    return graph.compile()


# --- Main Execution ---
async def extract_document_info(file_path: str, parser: str = "auto", multimodal: bool = False, verifier_model: str = None) -> ExtractionResult:
    # Override verifier model if specified
    if verifier_model:
        global VERIFIER_MODEL
        VERIFIER_MODEL = verifier_model
    
    print("🚀 Starting Document Extraction Workflow")
    print("=" * 60)
    print("   Pipeline: Classifier → Schema → Extractor → Verifier")
    if multimodal:
        print("   📷 Multimodal mode: ENABLED (using Gemini vision)")
    
    if parser == "pymupdf":
        content = parse_pdf_fallback(file_path)
    elif parser == "docling":
        content = parse_document_with_docling(file_path)
    else:
        content = parse_document_with_docling(file_path)
        if "GLYPH<" in content or len(content) < 1000:
            print("\n⚠️  Docling output appears garbled, trying PyMuPDF...")
            content = parse_pdf_fallback(file_path)
    
    page_images = []
    if multimodal:
        page_images = extract_pdf_pages_as_images(file_path)
    
    workflow = build_extraction_workflow()
    
    initial_state = {
        "document_path": file_path, "document_content": content, "page_images": page_images,
        "use_multimodal": multimodal and len(page_images) > 0,
        "document_type": None, "classification_confidence": None, "detected_language": None,
        "classification_reasoning": None, "schema_fields": None, "extraction_notes": None,
        "extracted_data": None, "verified_data": None, "verification_report": None, "messages": []
    }
    
    print("\n🔄 Running extraction pipeline...")
    final_state = await workflow.ainvoke(initial_state)
    
    # Determine final data: use verified if available, otherwise extracted
    extracted = final_state['extracted_data'] or {}
    verified = final_state.get('verified_data')
    final = verified if verified else extracted
    
    return ExtractionResult(
        document_type=final_state['document_type'] or "unknown",
        confidence=final_state['classification_confidence'] or 0.0,
        detected_language=final_state['detected_language'] or "unknown",
        schema_fields=final_state['schema_fields'] or [],
        extracted_data=extracted,
        verified_data=verified,
        verification_report=final_state.get('verification_report'),
        final_data=final,  # THE CORRECTED OUTPUT
        raw_text_preview=content[:500]
    )


def print_results(result: ExtractionResult):
    print("\n" + "=" * 60)
    print("📊 EXTRACTION RESULTS")
    print("=" * 60)
    
    print(f"\n📑 Document Classification:")
    print(f"   Type: {result.document_type}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Language: {result.detected_language}")
    
    print(f"\n📐 Dynamic Schema ({len(result.schema_fields)} fields):")
    for f in result.schema_fields[:8]:
        arabic = f" ({f.get('arabic_label')})" if f.get('arabic_label') else ""
        print(f"   • {f.get('name')}{arabic}")
    if len(result.schema_fields) > 8:
        print(f"   ... and {len(result.schema_fields) - 8} more")
    
    # Show final (corrected) data
    print(f"\n📋 Final Data (with corrections applied):")
    for key, value in result.final_data.items():
        if value is not None and value != "" and key != "raw_response":
            display = str(value)[:70] + "..." if len(str(value)) > 70 else str(value)
            print(f"   {key}: {display}")
    
    if result.verification_report:
        print(f"\n🔧 Corrections Applied ({len(result.verification_report)}):")
        for corr in result.verification_report:
            print(f"   • {corr.get('field_name')}: {corr.get('original_value')} → {corr.get('corrected_value')}")
    
    print("\n" + "-" * 60)


# --- Entry Point ---
if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Extract information from Arabic documents using a 4-agent pipeline.")
    arg_parser.add_argument("pdf_path", help="Path to the PDF document")
    arg_parser.add_argument("--output", "-o", help="Output JSON file")
    arg_parser.add_argument("--parser", choices=["docling", "pymupdf", "auto"], default="auto")
    arg_parser.add_argument("--multimodal", "-m", action="store_true", help="Use Gemini vision")
    arg_parser.add_argument("--verifier-model", "-v", default=None, 
                           help="Model for verification (default: gemini-2.5-flash-preview-05-20)")
    
    args = arg_parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"❌ Error: File not found: {args.pdf_path}")
        sys.exit(1)
    
    try:
        result = asyncio.run(extract_document_info(args.pdf_path, parser=args.parser, multimodal=args.multimodal, verifier_model=args.verifier_model))
        print_results(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
