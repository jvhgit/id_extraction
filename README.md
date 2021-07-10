# id_extraction
This repo provides the template id, the script to train and some instructions.

For this script to work one needs to install the packages (pip) used in the script. Also, one needs the Google OCR engine (Tesseract) + the Python wrapper (pip install pytesseract).
I used the 4.0 version of the engine (since it supported the Dutch language) and downloaded the engine from https://digi.bib.uni-mannheim.de/tesseract/. This will install the engine (one might needs to adjust some directories in the script). The code provided is not clean (since it is was an evening project) but I can clarify things if one asks me.

The code in the provided script is meant to be executed in an interactive session (that is how I used it). The steps in general:

1) Load in the template, annotate it (the boxes)
2) Based on this original example; generate more data by adding different paddings.
3) Make a CNN model (Keras API) to regress to the box coordinates
4) Use the extracted coordinates to cut out an image (name or date of birth)
5) Use the OCR engine to extract text from the image

This is done two times: one with horizontal ID cards and one with horizontal and vertical ID cards (to see how the CNN would perform with more 'variance' in the data).


Good luck!
