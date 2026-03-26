"""
PDF generation service using fpdf2.
"""
from io import BytesIO
from fpdf import FPDF
import re

def generate_pdf_from_markdown(markdown_text: str, title: str = "Research Limitations Report") -> BytesIO:
    """
    Generate a PDF from markdown text.
    
    Args:
        markdown_text: The markdown content to convert
        title: The title of the report
    
    Returns:
        BytesIO: PDF content in memory
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Add a title
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, title, align="C") # multi_cell to avoid crashes with long titles
    pdf.ln(5)
    
    # Parse markdown and add to PDF
    pdf.set_font("Arial", "", 12)
    
    # Simple markdown parsing
    lines = markdown_text.split('\n')
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                pdf.ln(2)
            else: 
                pdf.ln(5)
            continue
            
        # Headers
        if line.startswith('# '):
            in_list = False
            pdf.set_font("Arial", "B", 14)
            pdf.multi_cell(0, 10, line[2:])
            pdf.set_font("Arial", "", 12)
        elif line.startswith('## '):
            in_list = False
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 8, line[3:])
            pdf.set_font("Arial", "", 12)
        elif line.startswith('### '):
            in_list = False
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, line[4:])
            pdf.set_font("Arial", "", 12)
        # Lists
        elif line.startswith('- '):
            in_list = True
            pdf.set_font("Arial", "", 11)
            current_x = pdf.get_x()
            pdf.set_x(current_x + 10)
            pdf.multi_cell(0, 6, f"• {line[2:]}")
            pdf.set_x(current_x)
        # Regular text or Bold
        else:
            in_list = False
            pdf.set_font("Arial", "", 11)
            clean_line = line.replace('**', '')
            pdf.multi_cell(0, 6, clean_line)
    
    # Save to buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer
