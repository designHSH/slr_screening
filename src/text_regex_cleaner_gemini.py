import fitz  # PyMuPDF
import os
import re
from collections import Counter

def extract_text_by_page(pdf_path, skip_tables=True):
    """
    Extracts text from each page of a PDF file, with an option to skip table content.
    
    Args:
        pdf_path (str): The path to the PDF file.
        skip_tables (bool): If True, attempts to identify and exclude text within tables.
        
    Returns:
        list: A list of text content, one string per page.
    """
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page in doc:
            if not skip_tables:
                # Original, simple text extraction
                pages_text.append(page.get_text("text"))
                continue

            # --- Advanced extraction to skip tables ---
            # 1. Find the bounding boxes of all tables on the page and convert them to Rect objects
            table_rects = [fitz.Rect(tab.bbox) for tab in page.find_tables()]

            # 2. Get all text blocks on the page with their bounding boxes
            text_blocks = page.get_text("blocks")
            
            non_table_text = []
            for block in text_blocks:
                # The block format is (x0, y0, x1, y1, "text...", block_no, block_type)
                block_rect = fitz.Rect(block[:4])
                is_in_table = False
                
                # 3. Check if the text block is inside any of the table boxes
                for table_rect in table_rects:
                    if table_rect.contains(block_rect):
                        is_in_table = True
                        break
                
                # 4. If the block is not in a table, keep its text
                if not is_in_table:
                    non_table_text.append(block[4])
            
            pages_text.append("\n".join(non_table_text))

        doc.close()
        return pages_text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return []

def identify_headers_footers(pages, commonality_threshold=0.5):
    """
    Identifies common lines at the top and bottom of pages to be considered headers or footers.
    
    Args:
        pages (list): A list of text content, one string per page.
        commonality_threshold (float): Percentage of pages a line must appear on to be a header/footer.
        
    Returns:
        tuple: A tuple containing a set of header lines and a set of footer lines.
    """
    if not pages:
        return set(), set()
        
    num_pages = len(pages)
    if num_pages < 3: # Not enough pages to reliably detect headers/footers
        return set(), set()

    # --- Collect potential header and footer lines ---
    # We look at the first and last 3 lines of each page
    potential_headers = []
    potential_footers = []
    
    for text in pages:
        lines = text.strip().split('\n')
        if len(lines) > 1:
            potential_headers.extend(lines[:8])
            potential_footers.extend(lines[-8:])

    # --- Count the occurrences of these lines ---
    header_counts = Counter(line.strip() for line in potential_headers if line.strip())
    footer_counts = Counter(line.strip() for line in potential_footers if line.strip())

    # --- Identify lines that appear on a significant portion of pages ---
    required_occurrences = int(num_pages * commonality_threshold)
    
    # A line is a header if it appears frequently and is not just a number (page number)
    headers = {
        line for line, count in header_counts.items()
        if count >= required_occurrences and not line.strip().isdigit()
    }
    
    # A line is a footer if it appears frequently
    footers = {
        line for line, count in footer_counts.items()
        if count >= required_occurrences
    }
    
    return headers, footers

def remove_headers_footers_from_pages(pages, headers, footers):
    """Removes identified header and footer lines from each page."""
    cleaned_pages = []
    for text in pages:
        lines = text.strip().split('\n')
        
        # Filter out header and footer lines
        # This approach is more robust as it removes any line matching a known header/footer
        core_lines = [
            line for line in lines 
            if line.strip() not in headers and line.strip() not in footers
        ]
        
        cleaned_pages.append("\n".join(core_lines))
    return cleaned_pages

def clean_full_text(full_text):
    """
    Applies regex patterns to remove unwanted sections from the merged text.
    """
    
    # --- Group 1: Remove these sections and everything that follows ---
    # This is for sections that mark the end of the main content.
    end_of_paper_patterns = [
        r'^\s*references\s*$',
        r'^\s*bibliography\s*$',
        r'^\s*acknowledgements\s*$',
        r'^\s*acknowledgments\s*$',
        r'^\s*funding\s*$',
        r'^\s*appendix\s*$',
        r'^\s*supplementary material\s*$',
    ]
    
    # Combine patterns into a single regex for efficiency
    end_pattern_regex = re.compile(r'(' + '|'.join(end_of_paper_patterns) + r')', re.IGNORECASE | re.MULTILINE)
    
    match = end_pattern_regex.search(full_text)
    if match:
        # Keep everything before the match
        full_text = full_text[:match.start()]

    # --- Group 2: Remove just the specific section or line ---
    # This is for metadata that can appear anywhere.
    inline_removal_patterns = [
        r'^\s*author contributions.*$',
        r'^\s*affiliations\s*$',
        r'^\s*author information.*$',
        r'^\s*corresponding author.*$',
        r'.*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\s*$', # Removes lines with email addresses
        r'^\s*competing interests.*$',
        r'^\s*conflict of interest.*$'
    ]

    for pattern in inline_removal_patterns:
        full_text = re.sub(pattern, '', full_text, flags=re.IGNORECASE | re.MULTILINE)

    # --- Final cleanup ---
    # Remove excessive blank lines
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    return full_text.strip()


def process_paper(pdf_path):
    """
    Main processing pipeline for a single PDF file.
    """
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    # 1. Extract text page-by-page, skipping tables by default
    pages = extract_text_by_page(pdf_path, skip_tables=True)
    if not pages:
        return None # Skip if paper could not be read
        
    # 2. Detect headers/footers
    headers, footers = identify_headers_footers(pages)
    
    # 3. Remove headers/footers from each page
    cleaned_pages = remove_headers_footers_from_pages(pages, headers, footers)
    
    # 4. Merge cleaned pages into one long text
    full_text = "\n\n".join(cleaned_pages)
    
    # 5. Apply section removal patterns
    final_text = clean_full_text(full_text)
    
    return final_text

def main():
    """
    Main function to set up directories and process all PDFs.
    """
    # --- Configuration ---
    # Create dummy directories and files for demonstration
    pdf_directory = r"data\test_data\validation\manully_screened"
    output_directory = r"data\test_data\validation\claned_text"
    
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created a sample directory: '{pdf_directory}'")
        print("Please add your PDF files to this directory and run the script again.")
        # As an example, we can't create a real PDF, but we can show the structure.
        with open(os.path.join(pdf_directory, "placeholder.txt"), "w") as f:
            f.write("This is a placeholder. Put your PDFs here.")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # --- Processing Loop ---
    all_pdfs = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    if not all_pdfs:
        print(f"No PDF files found in '{pdf_directory}'.")
        return

    for pdf_file in all_pdfs:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        
        # Process the paper
        cleaned_text = process_paper(pdf_path)
        
        if cleaned_text:
            # Save the final cleaned text
            output_filename = os.path.splitext(pdf_file)[0] + '.txt'
            output_path = os.path.join(output_directory, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"Successfully saved cleaned text to: {output_path}\n")

if __name__ == '__main__':
    main()

