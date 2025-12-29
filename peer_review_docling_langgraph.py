"""
Peer Review Workflow using Docling + LangGraph (MULTIMODAL)

This implements the Reflection Pattern with:
- Producer Agent: Writes/improves the peer review
- Critic Agent: Provides constructive feedback on the review

Docling is used to parse research paper PDFs into structured text AND images.
LangGraph orchestrates the multi-agent reflection loop.
Gemini processes both text and figures for comprehensive review.

Install dependencies:
    uv add docling langgraph langchain-google-genai pillow

Usage:
    uv run peer_review_docling_langgraph.py path/to/paper.pdf
    uv run peer_review_docling_langgraph.py path/to/paper.pdf --multimodal
"""

import asyncio
import base64
import os
import sys
import io
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

# Docling imports for PDF parsing
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# LangChain imports for LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# --- Configuration ---
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env")
    sys.exit(1)
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = api_key

MAX_ITERATIONS = 3  # Maximum reflection iterations
MAX_IMAGES = 10      # Maximum number of images to include (to manage context window)

# --- State Definition ---
class PeerReviewState(TypedDict):
    """State for the peer review workflow."""
    paper_content: str           # Parsed paper content from Docling
    paper_title: str             # Paper title
    paper_images: list           # List of (image_bytes, caption) tuples
    multimodal: bool             # Whether to use multimodal mode
    current_review: str          # Current version of the peer review
    critic_feedback: str         # Feedback from the critic
    iteration: int               # Current iteration number
    is_approved: bool            # Whether the review is approved
    messages: Annotated[list, add_messages]  # Message history

# --- Document Parser using Docling ---
def parse_pdf_with_docling(pdf_path: str, extract_images: bool = False) -> tuple[str, str, list]:
    """
    Parse a PDF research paper using Docling.
    
    Args:
        pdf_path: Path to the PDF file
        extract_images: Whether to extract images for multimodal processing
    
    Returns:
        tuple: (paper_content, paper_title, images_list)
               images_list contains (image_bytes, caption) tuples
    """
    print(f"📄 Parsing PDF: {pdf_path}")
    print(f"   Image extraction: {'enabled' if extract_images else 'disabled'}")
    
    # Configure Docling pipeline options
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,  # Disable OCR for faster processing
        do_table_structure=True,  # Extract table structure
        generate_picture_images=extract_images,  # Extract figure images
        generate_table_images=extract_images,    # Extract table images
    )
    
    # Create format options for PDF
    pdf_format_option = PdfFormatOption(
        pipeline_options=pipeline_options
    )
    
    # Create converter with format options
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: pdf_format_option
        }
    )
    
    # Convert document
    result = converter.convert(pdf_path)
    
    # Extract text content
    document = result.document
    
    # Get title (usually first heading or from metadata)
    title = document.name or "Unknown Paper"
    
    # Export to markdown for better structure preservation
    content = result.document.export_to_markdown()
    
    # Extract images if multimodal mode
    images_list = []
    if extract_images:
        images_list = extract_images_from_document(document)
    
    print(f"✅ Parsed paper: {title}")
    print(f"   Content length: {len(content)} characters")
    print(f"   Images extracted: {len(images_list)}")
    
    return content, title, images_list


def extract_images_from_document(document) -> list:
    """
    Extract images from a Docling document using the pictures attribute.
    
    Docling stores extracted figures in document.pictures, each with:
    - picture.image.pil_image: PIL Image object
    - picture.caption_text(document): Caption string
    
    Returns:
        List of (image_bytes, caption) tuples
    """
    images = []
    
    if not hasattr(document, 'pictures'):
        print("   ⚠️ Document has no pictures attribute")
        return images
    
    for i, picture in enumerate(document.pictures):
        try:
            # Get PIL image from the ImageRef
            img_ref = picture.image
            if img_ref is None:
                continue
                
            # Access the PIL image directly
            pil_img = img_ref.pil_image
            if pil_img is None:
                continue
            
            # Convert PIL image to bytes (PNG format)
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            # Get caption using the caption_text method
            caption = ""
            if hasattr(picture, 'caption_text'):
                caption = picture.caption_text(document) or ""
            
            if not caption:
                caption = f"Figure {i+1}"
            
            images.append((img_bytes, caption))
            print(f"   📷 Extracted: {caption[:60]}..." if len(caption) > 60 else f"   📷 Extracted: {caption}")
            
        except Exception as e:
            print(f"   ⚠️ Could not extract picture {i+1}: {e}")
            continue
    
    return images


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

# --- LLM Setup ---
def get_llm():
    """Create the LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
    )

# --- Agent Prompts ---
PRODUCER_SYSTEM_PROMPT = """You are an expert academic peer reviewer. Your task is to write 
a comprehensive, constructive peer review of a research paper.

Your review should include:
1. **Summary**: Brief summary of the paper's main contributions
2. **Strengths**: What the paper does well (methodology, novelty, clarity)
3. **Weaknesses**: Areas that need improvement (gaps, unclear sections, missing details)
4. **Questions for Authors**: Specific questions that should be addressed
5. **Minor Comments**: Typos, formatting, citations issues
6. **Figures & Visuals** (if images provided): Evaluate clarity, labeling, and effectiveness
7. **Overall Assessment**: Your recommendation (Accept/Minor Revision/Major Revision/Reject)

Be specific, cite sections/pages/figures when possible, and maintain a professional, constructive tone.

When reviewing figures and images:
- Assess if figures effectively communicate the key results
- Check that axes are labeled, legends are clear, and resolution is adequate
- Note if any figures are redundant or if additional visualizations would help
- Comment on the quality of tables, diagrams, and architecture illustrations
"""

CRITIC_SYSTEM_PROMPT = """You are a meta-reviewer who evaluates peer reviews for quality.
Your task is to critique the peer review and suggest improvements.

Evaluate the review on:
1. **Completeness**: Does it cover all important aspects of the paper?
2. **Specificity**: Are criticisms backed by specific examples from the paper?
3. **Constructiveness**: Does it provide actionable feedback?
4. **Balance**: Does it acknowledge both strengths and weaknesses?
5. **Professionalism**: Is the tone appropriate for academic discourse?

If the review is excellent and comprehensive, respond with:
"APPROVED: This review is ready for submission."

Otherwise, provide specific suggestions for improvement.
"""

# --- Multimodal Message Builder ---
def build_multimodal_message(text_prompt: str, images: list, max_images: int = MAX_IMAGES) -> HumanMessage:
    """
    Build a multimodal HumanMessage with text and images.
    
    Args:
        text_prompt: The text content of the message
        images: List of (image_bytes, caption) tuples
        max_images: Maximum number of images to include
    
    Returns:
        HumanMessage with multimodal content
    """
    content = []
    
    # Add text first
    content.append({
        "type": "text",
        "text": text_prompt
    })
    
    # Add images (limited to max_images)
    images_to_include = images[:max_images]
    
    if images_to_include:
        content.append({
            "type": "text", 
            "text": f"\n\n**FIGURES AND IMAGES FROM THE PAPER** ({len(images_to_include)} images):\n"
        })
        
        for i, (img_bytes, caption) in enumerate(images_to_include):
            # Add caption
            caption_text = caption if caption else f"Figure {i+1}"
            content.append({
                "type": "text",
                "text": f"\n[{caption_text}]:"
            })
            
            # Add image as base64
            img_base64 = encode_image_to_base64(img_bytes)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
    
    return HumanMessage(content=content)


# --- Agent Nodes ---
def producer_node(state: PeerReviewState) -> PeerReviewState:
    """
    Producer agent: Writes or improves the peer review.
    Supports multimodal input (text + images) when enabled.
    """
    llm = get_llm()
    iteration = state.get("iteration", 0)
    multimodal = state.get("multimodal", False)
    images = state.get("paper_images", [])
    
    print(f"\n{'='*50}")
    print(f"📝 Producer Agent (Iteration {iteration + 1})")
    if multimodal and images:
        print(f"   🖼️  Multimodal mode: analyzing {len(images)} images")
    
    if iteration == 0:
        # First iteration: Write initial review
        base_prompt = f"""Please write a comprehensive peer review for the following research paper:

**Paper Title**: {state['paper_title']}

**Paper Content**:
{state['paper_content'][:15000]}"""  # Limit content length for context window
        
        if multimodal and images:
            base_prompt += """

**IMPORTANT**: I have included figures and images from the paper below. Please:
1. Analyze each figure for clarity, informativeness, and relevance
2. Comment on whether figures effectively support the text
3. Note any issues with figure quality, labeling, or presentation
4. Reference specific figures in your review when discussing methodology or results"""
        
        base_prompt += "\n\nWrite your peer review following the structured format."
        
    else:
        # Subsequent iterations: Improve based on feedback
        base_prompt = f"""Please improve your peer review based on the critic's feedback.

**Your Previous Review**:
{state['current_review']}

**Critic's Feedback**:
{state['critic_feedback']}

Please address all the points raised and provide an improved review."""
    
    # Build messages
    messages = [SystemMessage(content=PRODUCER_SYSTEM_PROMPT)]
    
    # Use multimodal message for first iteration if images available
    if iteration == 0 and multimodal and images:
        messages.append(build_multimodal_message(base_prompt, images))
    else:
        messages.append(HumanMessage(content=base_prompt))
    
    response = llm.invoke(messages)
    review = response.content
    
    print(f"   Review length: {len(review)} characters")
    
    return {
        **state,
        "current_review": review,
        "iteration": iteration + 1,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Producer Iteration {iteration + 1}]"),
            AIMessage(content=review[:500] + "...")  # Truncate for history
        ]
    }

def critic_node(state: PeerReviewState) -> PeerReviewState:
    """
    Critic agent: Evaluates the review and provides feedback.
    """
    llm = get_llm()
    
    print(f"\n{'='*50}")
    print(f"🔍 Critic Agent (Evaluating Iteration {state['iteration']})")
    
    prompt = f"""Please evaluate this peer review and provide feedback:

**Paper Title**: {state['paper_title']}

**Peer Review to Evaluate**:
{state['current_review']}

Remember: If the review is excellent and comprehensive, respond with:
"APPROVED: This review is ready for submission."

Otherwise, provide specific suggestions for improvement."""
    
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    feedback = response.content
    
    # Check if approved
    is_approved = "APPROVED:" in feedback.upper()
    
    print(f"   Feedback length: {len(feedback)} characters")
    print(f"   Status: {'✅ APPROVED' if is_approved else '🔄 Needs Improvement'}")
    
    return {
        **state,
        "critic_feedback": feedback,
        "is_approved": is_approved,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Critic Iteration {state['iteration']}]"),
            AIMessage(content=feedback[:500] + "...")
        ]
    }

# --- Routing Logic ---
def should_continue(state: PeerReviewState) -> Literal["producer", "end"]:
    """
    Determine if the reflection loop should continue.
    """
    if state.get("is_approved", False):
        print("\n✅ Review approved by critic!")
        return "end"
    
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        print(f"\n⚠️ Max iterations ({MAX_ITERATIONS}) reached.")
        return "end"
    
    print(f"\n🔄 Continuing to iteration {state.get('iteration', 0) + 1}...")
    return "producer"

# --- Build the Graph ---
def build_peer_review_graph():
    """
    Build the LangGraph workflow for peer review with reflection.
    """
    # Create the graph
    workflow = StateGraph(PeerReviewState)
    
    # Add nodes
    workflow.add_node("producer", producer_node)
    workflow.add_node("critic", critic_node)
    
    # Set entry point
    workflow.set_entry_point("producer")
    
    # Add edges
    workflow.add_edge("producer", "critic")  # Producer -> Critic
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "producer": "producer",  # Loop back for improvement
            "end": END               # Finish
        }
    )
    
    # Compile the graph
    return workflow.compile()

# --- Main Execution ---
async def run_peer_review(pdf_path: str, multimodal: bool = False):
    """
    Run the complete peer review workflow.
    
    Args:
        pdf_path: Path to the PDF file
        multimodal: If True, extract and analyze images from the paper
    """
    print("🚀 Starting Peer Review Workflow")
    if multimodal:
        print("🖼️  MULTIMODAL MODE: Will analyze figures and images")
    print("=" * 60)
    
    # Step 1: Parse the PDF (with image extraction if multimodal)
    try:
        paper_content, paper_title, images = parse_pdf_with_docling(
            pdf_path, 
            extract_images=multimodal
        )
    except Exception as e:
        print(f"❌ Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 2: Initialize state
    initial_state: PeerReviewState = {
        "paper_content": paper_content,
        "paper_title": paper_title,
        "paper_images": images,
        "multimodal": multimodal,
        "current_review": "",
        "critic_feedback": "",
        "iteration": 0,
        "is_approved": False,
        "messages": []
    }
    
    # Step 3: Build and run the graph
    graph = build_peer_review_graph()
    
    print("\n🔄 Starting reflection loop...")
    final_state = graph.invoke(initial_state)
    
    # Step 4: Output results
    print("\n" + "=" * 60)
    print("📋 FINAL PEER REVIEW")
    print("=" * 60)
    print(f"\nPaper: {final_state['paper_title']}")
    print(f"Iterations: {final_state['iteration']}")
    print(f"Multimodal: {'Yes' if final_state.get('multimodal') else 'No'}")
    print(f"Images analyzed: {len(final_state.get('paper_images', []))}")
    print(f"Approved: {'Yes' if final_state['is_approved'] else 'No (max iterations)'}")
    print("\n" + "-" * 60)
    print(final_state['current_review'])
    print("-" * 60)
    
    return final_state

# --- CLI Entry Point ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run peer_review_docling_langgraph.py <path_to_pdf> [--multimodal]")
        print("\nExamples:")
        print("  uv run peer_review_docling_langgraph.py research_paper.pdf")
        print("  uv run peer_review_docling_langgraph.py research_paper.pdf --multimodal")
        print("\nOptions:")
        print("  --multimodal    Extract and analyze figures/images from the paper")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    multimodal = "--multimodal" in sys.argv or "-m" in sys.argv
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    asyncio.run(run_peer_review(pdf_path, multimodal=True))

