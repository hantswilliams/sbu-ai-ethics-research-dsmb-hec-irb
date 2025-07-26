#!/usr/bin/env python3
"""
Combine Manuscript Sections

This script combines individual markdown files (abstract, intro, methods, results, discussion)
into a single consolidated manuscript file (_manuscript.md) and optionally converts it to PDF.

Usage:
    python combine_manuscript.py [--no-pdf] [--base-path PATH]
"""

import os
import argparse
import subprocess
from pathlib import Path
import logging
from datetime import datetime
import sys
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("manuscript_combiner")

def fix_image_paths(content, project_root):
    """
    Fix relative image paths to absolute paths so they work in the combined PDF.
    
    Args:
        content (str): Markdown content with relative image paths.
        project_root (Path): Path to the project root.
        
    Returns:
        str: Markdown content with fixed image paths.
    """
    # Define a pattern to match markdown image links
    # ![alt text](../path/to/image.png)
    image_pattern = r'!\[(.*?)\]\((\.\.\/.*?)\)'
    
    def replace_path(match):
        alt_text = match.group(1)
        rel_path = match.group(2)
        
        # Replace ../data/ with the absolute path
        if rel_path.startswith('../data/'):
            rel_path_parts = rel_path.split('/')
            # Remove the '..' to get a proper relative path from project root
            rel_path_parts = rel_path_parts[1:]  # Remove '../'
            abs_path = project_root.joinpath(*rel_path_parts)
            return f'![{alt_text}]({abs_path})'
        return match.group(0)  # Return unchanged if pattern doesn't match
    
    # Replace all image paths
    fixed_content = re.sub(image_pattern, replace_path, content)
    return fixed_content

def combine_markdown_files(base_path=None):
    """
    Combine markdown files into a single manuscript file.
    
    Args:
        base_path (str, optional): Base path to the project. Defaults to None.
        
    Returns:
        tuple: (bool, str) - Success status and path to the output file
    """
    if base_path:
        project_root = Path(base_path)
    else:
        # Assume script is in code/ directory
        script_path = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = script_path.parent
        
    manuscript_dir = project_root / "manuscript"
    
    # Ensure the manuscript directory exists
    if not manuscript_dir.exists():
        logger.error(f"Manuscript directory not found at {manuscript_dir}")
        return False, None
        
    # Define the data directories for image path resolution
    data_dir = project_root / "data"
        
    # Define the order of files to combine and their section titles
    sections = [
        {"file": "abstract.md", "title": None},  # No title for abstract as it's already added
        {"file": "intro.md", "title": "Introduction"},
        {"file": "methods.md", "title": "Methods"},
        {"file": "results.md", "title": "Results"},
        {"file": "discussion.md", "title": "Discussion"},
        {"file": "references.md", "title": "References"}
    ]
    
    # Create the output file
    output_file = manuscript_dir / "_manuscript.md"
    
    # Add header with generation timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    combined_content = f"""# WORKING DRAFT: Benchmarking Generative AI for Healthcare Ethics: A Comparative Analysis of Four Models Across Five Clinical Ethics Scenarios

**Version: {timestamp}**

## Abstract

"""
    
    # Process each file in order
    for section in sections:
        filename = section["file"]
        section_title = section["title"]
        file_path = manuscript_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Processing {filename}...")
        
        with open(file_path, 'r') as file:
            content = file.read()
            
            # Remove the first heading (# Title) as we want to use section headings instead
            lines = content.split('\n')
            if lines and lines[0].startswith('# '):
                content = '\n'.join(lines[1:])
            
            # Add section title if provided
            if section_title:
                combined_content += f"\n\n## {section_title}\n\n"
            
            # Add the content
            combined_content += content
    
    # Fix all image paths in the combined content
    combined_content = fix_image_paths(combined_content, project_root)
    
    # Write the combined content to the output file
    with open(output_file, 'w') as out_file:
        out_file.write(combined_content)
        
    logger.info(f"Combined manuscript created at {output_file}")
    return True, str(output_file)

def convert_markdown_to_pdf(markdown_file_path):
    """
    Convert a markdown file to PDF.
    
    Args:
        markdown_file_path (str): Path to the markdown file to convert.
        
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        markdown_path = Path(markdown_file_path)
        pdf_path = markdown_path.with_suffix('.pdf')
        
        # First try to use pandoc (preferred for academic documents with images)
        try:
            logger.info("Checking for pandoc installation...")
            subprocess.run(['pandoc', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Converting {markdown_path} to PDF using pandoc...")
            # Use pandoc with pdf-engine=xelatex for better Unicode and image support
            
            # Create a temporary LaTeX template for scientific styling
            latex_template = """\\documentclass[10pt,letterpaper]{article}
\\usepackage{lmodern}
\\usepackage{amssymb,amsmath}
\\usepackage{ifxetex,ifluatex}
\\usepackage{fontspec,xltxtra,xunicode}
\\defaultfontfeatures{Mapping=tex-text}
\\setromanfont[Mapping=tex-text]{Times New Roman}
\\setsansfont[Scale=MatchLowercase,Mapping=tex-text]{Arial}
\\setmonofont[Scale=MatchLowercase]{Consolas}
\\usepackage{sectsty}
\\allsectionsfont{\\sffamily\\bfseries}
\\usepackage{fancyhdr}
\\pagestyle{fancy}
\\pagenumbering{arabic}
\\usepackage{float}
\\floatplacement{figure}{H}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{graphicx}
\\usepackage{grffile}
\\usepackage[normalem]{ulem}
\\usepackage{hyperref}
\\hypersetup{breaklinks=true,
            bookmarks=true,
            colorlinks=true,
            citecolor=blue,
            urlcolor=blue,
            linkcolor=blue,
            pdfborder={0 0 0}}
\\urlstyle{same}
\\usepackage{caption}
\\usepackage{subcaption}
\\usepackage{setspace}
\\singlespacing
\\usepackage{geometry}
\\geometry{letterpaper,left=0.75in,right=0.75in,top=0.75in,bottom=0.75in}
\\setcounter{secnumdepth}{0}
\\title{$title$}
\\date{$date$}
\\begin{document}
\\maketitle
$body$
\\end{document}
"""
            
            # Write the LaTeX template to a file
            template_path = markdown_path.parent / "scientific_template.tex"
            with open(template_path, 'w') as template_file:
                template_file.write(latex_template)
            
            try:
                subprocess.run([
                    'pandoc',
                    str(markdown_path),
                    '-o', str(pdf_path),
                    '--pdf-engine=xelatex',
                    '--template', str(template_path),
                    '--variable', 'title=Benchmarking Generative AI for Healthcare Ethics: A Comparative Analysis of Four Models Across Five Clinical Ethics Scenarios',
                    '--variable', 'date=' + datetime.now().strftime('%B %d, %Y'),
                    '--standalone',
                    '--toc',  # Table of contents
                    '--number-sections'
                ], check=True)
                
                # Clean up template file
                if template_path.exists():
                    template_path.unlink()
                
                logger.info(f"PDF created successfully at {pdf_path}")
                return True
            except subprocess.SubprocessError as e:
                logger.warning(f"Error with xelatex: {e}, trying with pdflatex...")
                
                # Try with pdflatex instead
                try:
                    subprocess.run([
                        'pandoc',
                        str(markdown_path),
                        '-o', str(pdf_path),
                        '--pdf-engine=pdflatex',
                        '--variable', 'geometry:margin=0.75in',
                        '--variable', 'fontsize=10pt',
                        '--variable', 'linestretch=1.0',
                        '--standalone',
                        '--toc',  # Table of contents
                        '--number-sections'
                    ], check=True)
                    
                    # Clean up template file
                    if template_path.exists():
                        template_path.unlink()
                    
                    logger.info(f"PDF created successfully at {pdf_path}")
                    return True
                except subprocess.SubprocessError as e:
                    logger.warning(f"Error with pdflatex: {e}")
                    raise
            
            logger.info(f"PDF created successfully at {pdf_path}")
            return True
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Pandoc with wkhtmltopdf not available: {e}")
            
            # Try direct wkhtmltopdf with a temporary HTML file
            try:
                logger.info("Checking for wkhtmltopdf installation...")
                subprocess.run(['wkhtmltopdf', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Create a temporary HTML file from the markdown
                temp_html_path = markdown_path.with_suffix('.html')
                
                # Convert markdown to HTML using pandoc
                try:
                    logger.info("Converting markdown to HTML...")
                    subprocess.run([
                        'pandoc',
                        str(markdown_path),
                        '-o', str(temp_html_path),
                        '--standalone',
                        '--metadata', 'title=Manuscript'
                    ], check=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("Pandoc not available for HTML conversion, using simple HTML wrapper...")
                    
                    # Read markdown content
                    with open(markdown_path, 'r') as md_file:
                        md_content = md_file.read()
                    
                    # Create basic HTML with the markdown content and scientific styling
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Manuscript</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&family=Roboto:wght@300;400;500&display=swap');
        
        body {{
            font-family: 'Lora', 'Times New Roman', Times, serif;
            line-height: 1.5;
            margin: 20px;
            color: #333;
            font-size: 10pt;
            max-width: 95%;
            margin: 0 auto;
            padding: 1em;
            text-align: justify;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Roboto', Arial, sans-serif;
            font-weight: 500;
            color: #222;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
        }}
        
        h1 {{ font-size: 16pt; text-align: center; margin-top: 1em; }}
        h2 {{ font-size: 13pt; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }}
        h3 {{ font-size: 11pt; }}
        
        p {{ margin-bottom: 0.8em; }}
        
        img {{ 
            max-width: 90%; 
            height: auto; 
            display: block;
            margin: 1.5em auto;
            border: 1px solid #f0f0f0;
        }}
        
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin-bottom: 1.5em;
            font-size: 9pt;
        }}
        
        table, th, td {{ 
            border: 1px solid #ddd; 
        }}
        
        th, td {{ 
            padding: 6px 8px; 
            text-align: left; 
        }}
        
        th {{ 
            background-color: #f8f8f8;
            font-weight: 500;
            font-family: 'Roboto', Arial, sans-serif;
        }}
        
        tr:nth-child(even) {{ 
            background-color: #f9f9f9; 
        }}
        
        blockquote {{
            border-left: 4px solid #ddd;
            padding-left: 1em;
            color: #555;
            margin-left: 0;
            font-size: 9.5pt;
        }}
        
        code {{
            font-family: Consolas, Monaco, 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 9pt;
        }}
        
        hr {{
            border: 0;
            height: 1px;
            background: #ddd;
            margin: 1.5em 0;
        }}
        
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .caption {{
            text-align: center;
            font-style: italic;
            margin-top: -1.2em;
            margin-bottom: 1.5em;
            font-size: 9pt;
            color: #555;
        }}
        
        .footnotes {{
            border-top: 1px solid #ddd;
            padding-top: 0.8em;
            font-size: 9pt;
        }}
        
        /* Add page breaks before major sections */
        h2 {{
            page-break-before: always;
        }}
        
        /* Prevent orphan headings */
        h1, h2, h3, h4, h5, h6 {{
            page-break-after: avoid;
        }}
        
        /* Prevent images from breaking across pages */
        img {{
            page-break-inside: avoid;
        }}
    </style>
</head>
<body>
    <div id="content">
        {md_content}
    </div>
</body>
</html>"""
                    
                    # Write HTML to temp file
                    with open(temp_html_path, 'w') as html_file:
                        html_file.write(html_content)
                
                # Convert HTML to PDF using wkhtmltopdf
                logger.info(f"Converting HTML to PDF using wkhtmltopdf...")
                subprocess.run([
                    'wkhtmltopdf',
                    '--enable-local-file-access',
                    '--image-quality', '100',
                    '--margin-top', '10mm',
                    '--margin-right', '10mm',
                    '--margin-bottom', '10mm',
                    '--margin-left', '10mm',
                    '--page-size', 'Letter',
                    str(temp_html_path),
                    str(pdf_path)
                ], check=True)
                
                # Clean up temporary HTML file
                if temp_html_path.exists():
                    temp_html_path.unlink()
                
                logger.info(f"PDF created successfully at {pdf_path}")
                return True
                
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                logger.warning(f"wkhtmltopdf not available: {e}, falling back to markdown-pdf...")
                
                # Fall back to markdown-pdf
                # First, check if npm is available
                try:
                    subprocess.run(['npm', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.error("npm is not available. Please install Node.js and npm to use PDF conversion.")
                    return False
                    
                # Check if markdown-pdf is installed globally
                try:
                    subprocess.run(['markdown-pdf', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    md_pdf_installed = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    md_pdf_installed = False
                    
                # Install markdown-pdf if not already installed
                if not md_pdf_installed:
                    logger.info("Installing markdown-pdf npm package...")
                    subprocess.run(['npm', 'install', '-g', 'markdown-pdf'], check=True)
                
                # Create a CSS file for styling the PDF
                css_content = """
                @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&family=Roboto:wght@300;400;500&display=swap');
                
                body {
                    font-family: 'Lora', 'Times New Roman', Times, serif;
                    line-height: 1.5;
                    color: #333;
                    font-size: 10pt;
                    max-width: 95%;
                    margin: 0 auto;
                    padding: 1em;
                    text-align: justify;
                }
                
                h1, h2, h3, h4, h5, h6 {
                    font-family: 'Roboto', Arial, sans-serif;
                    font-weight: 500;
                    color: #222;
                    margin-top: 1.2em;
                    margin-bottom: 0.6em;
                }
                
                h1 { font-size: 16pt; text-align: center; margin-top: 1em; }
                h2 { font-size: 13pt; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }
                h3 { font-size: 11pt; }
                
                p { margin-bottom: 0.8em; }
                
                img { 
                    max-width: 90%; 
                    height: auto; 
                    display: block;
                    margin: 1.5em auto;
                    border: 1px solid #f0f0f0;
                }
                
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin-bottom: 1.5em;
                    font-size: 9pt;
                }
                
                table, th, td { 
                    border: 1px solid #ddd; 
                }
                
                th, td { 
                    padding: 6px 8px; 
                    text-align: left; 
                }
                
                th { 
                    background-color: #f8f8f8;
                    font-weight: 500;
                    font-family: 'Roboto', Arial, sans-serif;
                }
                
                tr:nth-child(even) { 
                    background-color: #f9f9f9; 
                }
                
                blockquote {
                    border-left: 4px solid #ddd;
                    padding-left: 1em;
                    color: #555;
                    margin-left: 0;
                    font-size: 9.5pt;
                }
                
                code {
                    font-family: Consolas, Monaco, 'Courier New', monospace;
                    background-color: #f5f5f5;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-size: 9pt;
                }
                
                hr {
                    border: 0;
                    height: 1px;
                    background: #ddd;
                    margin: 1.5em 0;
                }
                
                a {
                    color: #0366d6;
                    text-decoration: none;
                }
                
                a:hover {
                    text-decoration: underline;
                }
                """
                
                css_path = markdown_path.parent / "scientific_style.css"
                with open(css_path, 'w') as css_file:
                    css_file.write(css_content)
                
                logger.info(f"Converting {markdown_path} to PDF using markdown-pdf with scientific styling...")
                # markdown-pdf has more limited CLI options than we attempted to use
                # Let's use the simpler version that we know works
                subprocess.run([
                    'markdown-pdf',
                    str(markdown_path),
                    '-o', str(pdf_path),
                    '-s', str(css_path),
                    '--paper-format', 'Letter'
                ], check=True)
                
                # Clean up CSS file
                if css_path.exists():
                    css_path.unlink()
                
                logger.info(f"PDF created successfully at {pdf_path}")
                return True
                
    except subprocess.SubprocessError as e:
        logger.error(f"Error during PDF conversion: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during PDF conversion: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Combine markdown files into a single manuscript and convert to PDF')
    parser.add_argument('--base-path', type=str, default=None,
                        help='Base path to the project')
    parser.add_argument('--no-pdf', action='store_true',
                        help='Skip PDF conversion')
    parser.add_argument('--install-converters', action='store_true',
                        help='Provide instructions for installing PDF converters')
    args = parser.parse_args()
    
    if args.install_converters:
        print("\nPDF Converter Installation Instructions:")
        print("=======================================")
        print("\n1. Pandoc with wkhtmltopdf (recommended):")
        print("   - Install pandoc: https://pandoc.org/installing.html")
        print("   - Install wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        print("\n2. wkhtmltopdf alone:")
        print("   - Install wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        print("\n3. Markdown-pdf (fallback):")
        print("   - Install Node.js and npm: https://nodejs.org/")
        print("   - Install markdown-pdf globally: npm install -g markdown-pdf")
        print("\nThe script will try these methods in order and use the first one available.")
        return
    
    try:
        success, output_file = combine_markdown_files(base_path=args.base_path)
        if not success:
            logger.error("Failed to combine manuscript files")
            sys.exit(1)
            
        logger.info("Manuscript combination completed successfully")
        
        # Convert to PDF if requested
        if not args.no_pdf:
            if output_file:
                pdf_success = convert_markdown_to_pdf(output_file)
                if pdf_success:
                    logger.info("PDF conversion completed successfully")
                else:
                    logger.error("PDF conversion failed - try installing additional PDF converters")
                    logger.error("Run with --install-converters for installation instructions")
                    sys.exit(2)
            else:
                logger.error("Cannot convert to PDF: No output file path returned")
                sys.exit(3)
    except Exception as e:
        logger.error(f"Error processing manuscript: {e}")
        sys.exit(4)

if __name__ == "__main__":
    main()
