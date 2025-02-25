import os
import math

def make_tex(directory_path, output_file="presentation.tex"):
    """
    Generate a LaTeX beamer presentation with animated frames from a directory of images.
    
    Args:
        directory_path (str): Path to the directory containing the annotated images
        output_file (str): Path to save the output LaTeX file
    """
    # Count the number of image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.startswith("annotated_") and f.endswith(".png")]
    num_images = len(image_files)
    
    if num_images == 0:
        print(f"No annotated images found in {directory_path}")
        return
    
    # Find the highest image number to ensure we don't exceed it
    max_image_num = -1
    for f in image_files:
        try:
            num = int(f.replace("annotated_", "").replace(".png", ""))
            max_image_num = max(max_image_num, num)
        except ValueError:
            continue
    
    # Calculate the number of frames needed (500 images per frame)
    num_frames = math.ceil(max_image_num / 500)
    
    # Get the directory name for use in the LaTeX file
    dir_name = os.path.basename(os.path.normpath(directory_path))
    
    # Create the LaTeX preamble
    preamble = r"""\documentclass[pdf]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{gensymb}
\usepackage{amsmath,amsfonts}
\usepackage{graphicx}
\usepackage{pgfplots}
\usepackage{tikz}
\usepackage{mathtools}
\usepackage{amsthm}
\usetikzlibrary{arrows}
\usepackage{nccmath}
\usepackage[percent]{overpic}
\usepackage{rotating}
\usepackage{array, booktabs}
\usepackage{centernot}
\usepackage{animate}
\usepackage[absolute,overlay]{textpos}
% Begin tikz
\usepackage{pgf}
\usetikzlibrary{shadows,arrows,decorations,decorations.shapes,backgrounds,shapes,snakes,automata,fit,petri,shapes.multipart,calc,positioning,shapes.geometric,graphs,graphs.standard}
% End tikz
% for checkmark and xmarks
\usepackage{pifont}% http://ctan.org/pkg/pifont
\newcommand{\cmark}{\ding{51}}%
\newcommand{\xmark}{\ding{55}}%
\newcommand\figscale{0.74}
\mode<presentation>{}
\setlength{\fboxrule}{2pt}
%gets rid of bottom navigation symbols
\beamertemplatenavigationsymbolsempty
\setbeamertemplate{page number in head/foot}{}
%gets rid of footer
%will override 'frame number' instruction above
%comment out to revert to previous/default definitions
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\E}{\mathbb{E}}
\newcommand{\bbrpos}{\mathbb{R}_{\ge 0}}
\newcommand\bbr{\mathbb{R}}
\newcommand{\todo}[1]{{\color{red} todo: #1}}
\newcommand\fps{8}

\begin{document}
"""
    
    # Create the document body with frames
    body = ""
    for i in range(num_frames):
        start_idx = i * 500
        end_idx = min((i + 1) * 500, max_image_num)
        
        # Skip this frame if start_idx exceeds our max image number
        if start_idx > max_image_num:
            break
            
        frame = f"""
\\begin{{frame}}
\\begin{{center}}
\\animategraphics[loop,controls,autoplay,width=2.7 in]{{\\fps}}{{{dir_name}/annotated_}}{{{start_idx}}}{{{end_idx}}}
\\end{{center}}
\\end{{frame}}
"""
        body += frame
    
    # Add document closing
    closing = r"\end{document}"
    
    # Combine all parts
    full_document = preamble + body + closing
    
    # Write to output file
    with open(output_file, "w") as f:
        f.write(full_document)
    
    print(f"LaTeX file generated: {output_file}")
    print(f"Found {max_image_num + 1} images, created {num_frames} animation frames")
