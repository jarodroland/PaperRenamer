#! /opt/homebrew/Caskroom/miniforge/base/envs/ollama/bin/python

import argparse
import os
import sys
import ollama
from google import genai
import shutil
import re
import time
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, FloatObject, NameObject
from pathlib import Path
import tempfile

pathProjectRoot = os.path.join(Path.home(), 'projects', 'PaperRenamer')
pathInbox = os.path.join(pathProjectRoot, 'PaperRenamerInbox')
pathOutbox = os.path.join(pathProjectRoot, 'PaperRenamerOutbox')
pathLogs = os.path.join(pathProjectRoot, 'logs')

def fix_pdf_view_preferences(pdf_path):
    """
    Fixes PDF view preferences to force the PDF opens with the Single Page Continues Scrolling view and without side panels. 
    This is done by creating a new PDF with the desired view preferences and copying the content from the original PDF.
    Creating a new PDF also removes PDF/A status that prevents saving.
    """

    reader = PdfReader(pdf_path, strict=False)                  # set strict=False to be more forgiving of non-standard PDFs
    writer = PdfWriter()

    # create a writer and copy pages from the reader
    try:
        writer.append_pages_from_reader(reader)                 # try appending pages from the reader to and setting our own view preferences
    except AttributeError as e:
        print(f"Error occurred while appending pages: {e}, will try removing links and annotations which can cause issues with some PDFs.")
        try:
            writer_temp = PdfWriter(clone_from=reader)          # if append_pages_from_reader fails, remove links and annotations and try again
            writer_temp.remove_links()
            writer_temp.remove_annotations(subtypes=None)
            writer = PdfWriter()
            writer.append_pages_from_reader(writer_temp)
        except Exception as e2:
            print(f"Error still occurred after removing links and annotations: {e2}, will try cloning the reader which preserves the original structure and metadata.")
            try:
                writer = PdfWriter(clone_from=reader)           # if append_pages_from_reader fails, try cloning the reader in which case our new prefernces to fail to be applied but at least we can preserve the original PDF structure and metadata without risking further corruption by trying to modify it
            except Exception as e3:
                print(f"Error occurred while cloning reader: {e3}, unable to fix PDF view preferences.")
                return

    # build a /FitH open-action that fires when the PDF is opened to set the zoom to fit page width, and set the page layout to single page continuous scrolling and hide side panels like the bookmark pane
    first_page_ref = writer.pages[0].indirect_reference
    top_y = float(writer.pages[0].mediabox.top)
    writer._root_object.update({
        NameObject("/OpenAction"): ArrayObject([
            first_page_ref,
            NameObject("/FitH"),
            FloatObject(top_y),
        ]),                                                     # [page_ref, /FitH, top_y] → fits the page width to the window
        NameObject("/PageMode"): NameObject("/UseNone"),        # hide side panels like the Bookmark pane
        NameObject("/PageLayout"): NameObject("/OneColumn")     # set single page continuous scrolling 
        # NameObject("/PageLayout"): NameObject("/SinglePage")     # set single page continuous scrolling 
    })

    # write to a temp file, then atomically replace the original
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        writer.write(tmp)
    os.replace(tmp_path, new_filepath)


def extract_front_page_text(filepath):
    """Extracts text from the first page of the PDF, where metadata lives."""
    try:
        reader = PdfReader(filepath)
        # We usually only need the first page to find Title, Author, Journal, Year
        text = reader.pages[0].extract_text() 

        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def generate_filename_with_gemini(pathPdfFile):
    '''Feeds the PDF to Gemini API to generate the filename.'''

    pathApiKey = os.path.join(Path.home(), '.gemini', 'apikey-default.txt')
    try:
        with open(pathApiKey, "r") as fileApiKey:
            API_KEY = fileApiKey.read().strip()
    except FileNotFoundError:
        print(f'Error: API key file not found: {pathApiKey}. Please create a file named with your API key.')
        sys.exit(1)
    
    client = genai.Client(api_key=API_KEY)
    
    prompt = r"""
    You are a highly precise metadata extraction assistant. 
    Review this academic manuscript and extract:
    1. First Author's Last Name
    2. Publication Year
    3. Journal Title
    4. Manuscript Title

    Output EXACTLY a single string formatted as: "LastName - Year - JournalTitle - Manuscript Title"
    Rules:
    - Replace any colons in the output with dashes to ensure filesystem compatibility, but don't add any whitespaces when replacing.
    - Remove any other special characters (e.g., ? / \ < > | * ").
    - Make the output as concise as possible while remaining readable.
    - Do not include the .pdf extension in your output.
    - Provide ONLY the requested string, with no additional text, markdown, or explanation.
    """
    
    # print(f"Analyzing {os.path.basename(pdf_path)} with Gemini...")
    
    # upload the PDF to Gemini
    document = client.files.upload(file=pathPdfFile)
    
    # generate the new name based on the prompt
    response = client.models.generate_content(
        # model='gemini-flash-latest',
        model='gemini-flash-lite-latest',
        contents=[prompt, document]
    )

    new_name = response.text.strip()
    
    # clean up the file from Google's servers to save space
    client.files.delete(name=document.name)
    
    return new_name

def generate_filename_with_gemma(pathPdfFile):
    """Feeds the extracted text to local Gemma to generate the filename."""

    try:
        reader = PdfReader(pathPdfFile)
        text = reader.pages[0].extract_text()       # we usually only need the first page to find Title, Author, Journal, Year
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
     
    prompt = f"""
    You are a precise data extraction algorithm. Review the following text extracted from the first page of an academic manuscript.
    
    Extract:
    1. The last name of the First Author
    2. Publication Year
    3. Journal Title 
    4. Manuscript Title 

    Format: \"LastName - Year - Journal Title - Manuscript Title\"
    
    CRITICAL RULES:
    - Output EXACTLY and ONLY the formatted string.
    - Do not say "Here is the filename" or "Okay". 
    - Do not add `.pdf`.
    - If any text is all caps, convert it to title case (e.g. "NATURE" should become "Nature").
    
    Manuscript Text:
    ---
    {text} 
    ---
    """

    try:
        response = ollama.chat(
            model='gemma4:e4b',
            messages=[{'role': 'user', 'content': prompt}],
            keep_alive='4h'         # keep the model loaded for 4 hours to speed up subsequent calls by skipping the cold start loading time
        )
        
        new_name = response['message']['content'].strip()
        
        return new_name

    except Exception as e:
        print(f"Error communicating with local Gemma: {e}")
        return None


def rename_manuscript(filepath, model_to_use=None):
    if not filepath.endswith('.pdf'):
        print(f"Skipping {filepath} - not a PDF.")
        return

    # print(f"Reading: {filepath}...")


    if model_to_use == 'gemma':
        new_basename = generate_filename_with_gemma(filepath)
    elif model_to_use == 'gemini':
        new_basename = generate_filename_with_gemini(filepath)
    else:
        # if the computer is an apple silicon use the local Gemma model, otherwise use the Gemini API
        systemInfo = os.uname() 
        if (systemInfo.sysname == "Darwin" and systemInfo.machine == "arm64") or \
           (systemInfo.sysname == "Linux" and systemInfo.nodename == 'RolandLab'):
            new_basename = generate_filename_with_gemma(filepath)
        else:
            new_basename = generate_filename_with_gemini(filepath)

    if new_basename:
        new_basename = new_basename.replace(":", "-").replace(";", "-")             # replace any colon or semicolon with a dash to avoid filesystem issues
        new_basename = re.sub(r'[`!@#$%^&*()+=\/?:"<>|]', "", new_basename)         # remove any other special characters that are not allowed in filenames
        new_filename = f"{new_basename}.pdf"
        new_filepath = os.path.join(pathOutbox, new_filename)
        
        try:
            shutil.move(filepath, new_filepath)
            # print(f"✅ Success! Renamed to: {new_filename}")
        except Exception as e:
            print(f"❌ Error renaming file: {e}")
        
        return new_filepath
    else:
        print("❌ Failed to generate a new filename.")
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename a PDF manuscript.")
    parser.add_argument('-p', '--pdfPath', type=str, help="Path to the PDF file")
    parser.add_argument("-m", "--model", choices=['gemma', 'gemini'], help="Model to use for renaming (gemma or gemini)")
    args = parser.parse_args()
    
    # write a log message to a file to indicate the script has started
    os.makedirs(pathLogs, exist_ok=True)  # ensure the logs directory exists
    pathLogFile = os.path.join(pathLogs, 'paper_renamer.log')
    with open(pathLogFile, "a") as log_file:
        log_file.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting PaperRenamer script...\n")     # write a timestamp as YYYY-MM-DD HH:MM:SS
        log_file.write(f"Received arguments: {sys.argv}\n")
        
    # if a pdfPath argument was provided use it, otherwise look for the most recently added PDF in the inbox directory
    if not args.pdfPath:
        pdfFileList = [filePdf for filePdf in os.listdir(pathInbox) if filePdf.endswith('.pdf')]
        if not pdfFileList:
            print(f"No PDF files found in the inbox directory: {pathInbox}")
            sys.exit(1)
        latest_pdf = max(pdfFileList, key=lambda f: os.path.getctime(os.path.join(pathInbox, f)))
        args.pdfPath = os.path.join(pathInbox, latest_pdf)
        print(f"No PDF path provided, using the most recently added PDF in the inbox: {args.pdfPath}")

    new_filepath = rename_manuscript(args.pdfPath, model_to_use=args.model)
    if new_filepath and new_filepath != -1:
        fix_pdf_view_preferences(new_filepath)
        print(new_filepath)     # return new_filepath as the output of the script for use in scripting

        # display a dialog box to prompt the user to select a directory to move the renamed file to, with the default directory set ~/Dropbox/Research
        import tkinter as tk
        from tkinter import filedialog
        tkRootWindow = tk.Tk()
        tkRootWindow.withdraw()
        tkRootWindow.attributes("-topmost", True)  # make the dialog appear on top of other windows
        default_directory = os.path.join(Path.home(), 'Dropbox', 'Research')

        selected_directory = filedialog.askdirectory(title="Select a directory to move the renamed file to", initialdir=default_directory)
        if selected_directory:
            try:
                shutil.move(new_filepath, os.path.join(selected_directory, os.path.basename(new_filepath)))
            except Exception as e:
                print(f"Error moving file: {e}")
        else:
            print("No directory selected.")

        tkRootWindow.destroy()
    else:
        print("Error: Failed to rename the manuscript.")
        sys.exit(-1)
