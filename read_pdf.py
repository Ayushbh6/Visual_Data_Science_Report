#!/usr/bin/env python3
"""Script to extract text from PDF and analyze it"""

import sys
import os


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using available libraries"""
    text = ""

    # Try PyPDF2 first
    try:
        import PyPDF2

        print("Using PyPDF2 to extract text...")
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        print(f"PyPDF2 failed: {e}")

    # Try pypdf
    try:
        import pypdf

        print("Using pypdf to extract text...")
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if isinstance(page_text, str):
                    text += page_text + "\n\n"
                else:
                    text += str(page_text) + "\n\n"
        return text
    except Exception as e:
        print(f"pypdf failed: {e}")

    # Try PyMuPDF (fitz)
    try:
        import fitz

        print("Using PyMuPDF (fitz) to extract text...")
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n\n"
        doc.close()
        return text
    except Exception as e:
        print(f"PyMuPDF failed: {e}")

    raise Exception("All PDF extraction methods failed")


def main():
    pdf_path = os.path.join(
        os.path.dirname(__file__),
        "Docs",
        "186.868 Visual Data Science 2025W - Wrangle & Profile_ Attempt review _ TUWEL.pdf",
    )

    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return

    print(f"Extracting text from: {pdf_path}")

    try:
        text = extract_text_from_pdf(pdf_path)

        # Save extracted text to a file for easier review
        output_path = os.path.join(
            os.path.dirname(__file__), "Docs", "submitted_pdf_text.txt"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"\n=== PDF TEXT EXTRACTED (first 5000 chars) ===\n")
        print(text[:5000])
        print(f"\n=== Total characters: {len(text)} ===")
        print(f"\nFull text saved to: {output_path}")

        # Also count pages
        import PyPDF2

        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            print(f"Number of pages: {len(reader.pages)}")

    except Exception as e:
        print(f"Error extracting text: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
