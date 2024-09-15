from io import BytesIO

import markdown2

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from bs4 import BeautifulSoup

def convert_to_pdf(markdown_text):
    """
    Converts markdown text to a PDF document using ReportLab.
    """

    # Convert markdown to HTML
    html_text = markdown2.markdown(markdown_text)
    
    # Parse the HTML to plain text while keeping the tags
    soup = BeautifulSoup(html_text, "html.parser")

    # Create a ReportLab document
    pdf_output = BytesIO()
    doc = SimpleDocTemplate(pdf_output, pagesize=letter)

    # Define ReportLab styles
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        name="Custom",
        alignment=TA_LEFT,
        fontSize=12,
        leading=14,  # Space between lines
        spaceAfter=10,  # Space after each paragraph
        preserveWhiteSpace=True  # Preserve spaces
    )

    # Create a list to hold the content
    story = []

    # Convert each part of the HTML content to a ReportLab Paragraph
    for element in soup:
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            style = styles["Heading1"] if element.name == "h1" else styles["Heading2"]
        else:
            style = custom_style
        text = str(element)

        # Handle multiple newlines by adding Spacer elements
        if element.name == 'br':
            story.append(Spacer(1, 12))
        else:
            paragraphs = text.split('\n')
            for para in paragraphs:
                if para.strip():  # If the paragraph has content
                    story.append(Paragraph(para, style))
                else:
                    story.append(Spacer(1, 12))  # Add space for empty lines

    # Build the PDF document
    doc.build(story)

    # Move the pointer to the start of the file
    pdf_output.seek(0)

    return pdf_output