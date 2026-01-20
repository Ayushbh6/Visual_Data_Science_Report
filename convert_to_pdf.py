
import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md_path, output_pdf_path):
    # Read Markdown
    with open(source_md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert to HTML
    html_text = markdown.markdown(text)
    
    # Add some basic styling for better PDF look
    styled_html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Helvetica, sans-serif; font-size: 12pt; line-height: 1.5; }}
        h1 {{ font-size: 18pt; color: #333366; margin-top: 20px; margin-bottom: 10px; }}
        h2 {{ font-size: 16pt; color: #333366; margin-top: 15px; margin-bottom: 10px; border-bottom: 1px solid #ccc; }}
        h3 {{ font-size: 14pt; color: #333366; margin-top: 10px; }}
        p {{ margin-bottom: 10px; }}
        ul {{ margin-bottom: 10px; }}
        li {{ margin-bottom: 5px; }}
    </style>
    </head>
    <body>
    {html_text}
    </body>
    </html>
    """
    
    # Write to PDF
    with open(output_pdf_path, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            styled_html,                # the HTML to convert
            dest=result_file            # file handle to recieve result
        )
    
    return pisa_status.err

if __name__ == "__main__":
    source = "Docs/report_draft_phase_1.md"
    output = "Docs/Project_Report_Phase1.pdf"
    
    if convert_md_to_pdf(source, output):
        print("pisa convert err")
    else:
        print(f"Successfully created {output}")

