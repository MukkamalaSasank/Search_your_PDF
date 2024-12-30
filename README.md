# Search_your_PDF

This project provides a user-friendly Streamlit application that leverages Generative AI to answer your questions about a specified PDF document.

Requirements:

Python 3.10
Streamlit (pip install streamlit)
Transformers (pip install transformers)
Langchain Community (pip install langchain-community)
Sentence Transformers (pip install sentence-transformers)
Installation:

Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate.bat


Install dependencies:

Bash

pip install -r requirements.txt


Clone the model repository (if applicable):

If you're using a custom pre-trained model, clone its repository and follow the instructions to download the model files. You may need to adjust the checkpoint variable in app.py to point to the correct location.

Add your PDF document:
Place the PDF document you want to analyze in the "docs" folder within the project directory.

Running the App:

Start the Streamlit app:

Bash

streamlit run app.py


Open a web browser:

Visit http://localhost:8501 to access the interactive interface.