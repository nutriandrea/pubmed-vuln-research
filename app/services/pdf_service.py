"""
PDF generation service using fpdf2.
Handles complex markdown-style text and long titles without layout crashes.
"""
from io import BytesIO
from fpdf import FPDF
import re

def generate_pdf_from_markdown(markdown_text: str, title: str = "Research Report") -> BytesIO:
    """
    Convert a markdown string into a formatted PDF document.
    
    Args:
        markdown_text (str): The report content in markdown.
        title (str): The report title (often containing the long PubMed query).
        
    Returns:
        BytesIO: A memory buffer containing the generated PDF.
    """
    # Initialize FPDF with explicit units (mm) and A4 format
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    
    # Set explicit margins to avoid "out of bounds" calculation errors
    # Left, Top, Right = 20mm
    pdf.set_margins(left=20, top=20, right=20)
    pdf.add_page()
    
    # Calculate fixed width (A4 210mm - 20mm left - 20mm right = 170mm)
    # Using a fixed width instead of '0' prevents the "not enough horizontal space" exception
    effective_width = 170 

    # --- TITLE SECTION ---
    pdf.set_font("Arial", "B", 16)
    # multi_cell automatically wraps text if the title (or query) is too long
    pdf.multi_cell(w=effective_width, h=10, txt=title, align="C")
    pdf.ln(10)
    
    # Default font for body text
    pdf.set_font("Arial", "", 11)
    lines = markdown_text.split('\n')
    
    for line in lines:
        # FPDF standard fonts (Arial) only support Latin-1 encoding.
        # We strip non-compatible characters (emojis, special symbols) to avoid crashes.
        line = line.encode('latin-1', 'ignore').decode('latin-1').strip()
        
        if not line:
            pdf.ln(4) # Add small spacing for empty lines
            continue
            
        # Force cursor reset to the left margin before rendering each line.
        # This prevents cumulative X-offset errors that trigger the "no space" crash.
        pdf.set_x(20)

        # Handle Markdown Headers (#, ##, ###)
        if line.startswith('#'):
            # Determine header level by counting leading '#'
            header_level = len(line.split()[0]) 
            clean_text = line.lstrip('#').strip()
            
            # Adjust font size based on header level
            font_size = 14 if header_level == 1 else 12
            pdf.set_font("Arial", "B", font_size)
            pdf.multi_cell(w=effective_width, h=9, txt=clean_text)
            pdf.set_font("Arial", "", 11) # Reset to body font
            
        # Handle Unordered Lists (- or *)
        elif line.startswith('- ') or line.startswith('* '):
            # Replace markdown symbol with a standard bullet point
            clean_text = "- " + line[2:].strip()
            pdf.multi_cell(w=effective_width, h=6, txt=clean_text)
            
        # Handle Regular Paragraphs
        else:
            # Note: Complex inline bolding (**text**) is stripped to maintain 
            # layout stability and prevent horizontal calculation overflows.
            clean_text = line.replace('**', '').replace('*', '')
            pdf.multi_cell(w=effective_width, h=6, txt=clean_text)
            
    # Save the PDF document to an in-memory buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    
    return buffer
