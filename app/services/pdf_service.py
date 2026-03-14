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
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)
    
    # Parse markdown and add to PDF
    pdf.set_font("Arial", "", 12)
    
    # Simple markdown parsing
    lines = markdown_text.split('\n')
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
            
        # Headers
        if line.startswith('# '):
            if in_list:
                pdf.ln(5)
                in_list = False
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, line[2:], ln=True)
            pdf.set_font("Arial", "", 12)
        elif line.startswith('## '):
            if in_list:
                pdf.ln(5)
                in_list = False
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, line[3:], ln=True)
            pdf.set_font("Arial", "", 12)
        elif line.startswith('### '):
            if in_list:
                pdf.ln(5)
                in_list = False
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, line[4:], ln=True)
            pdf.set_font("Arial", "", 12)
        # Lists
        elif line.startswith('- '):
            if not in_list:
                in_list = True
            pdf.set_font("Arial", "", 11)
            pdf.cell(10, 6, "")
            pdf.multi_cell(0, 6, line[2:])
        # Bold text
        elif '**' in line:
            if in_list:
                pdf.ln(3)
                in_list = False
            # Simple bold replacement
            parts = re.split(r'\*\*(.*?)\*\*', line)
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    pdf.set_font("Arial", "B", 11)
                else:
                    pdf.set_font("Arial", "", 11)
                pdf.cell(0, 6, part, ln=True if i == len(parts) - 1 else False)
            if len(parts) == 1:
                pdf.ln(6)
        # Regular text
        else:
            if in_list:
                pdf.ln(3)
                in_list = False
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, line)
    
    # Save to buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer
