## How to run
Run the scripts/download_model.py to download the all-MiniLM-L6-v2 model. This process can take 10-20 seconds. <br>
Run shortcut.py. <br>
Paste your email sample or use an example from test.txt. Type "END" to run the model.

## Abstract
The Template Suggester is a feasibility prototype designed to assist in selecting suitable reply templates for recurring email inquiries. The core concept is sentence-embedded template matching: the system compares incoming emails against a library of pre-written templates to identify the most semantically similar response.
In the current iteration, templates are stored in a JSON structure containing titles, descriptions, keywords, and body text. The system encodes these templates into vector embeddings. When a user inputs an email, it is encoded into the same vector space, and the system calculates cosine similarity (implemented as a dot product after normalization) to rank the templates. The application then outputs the body of the best-matching template for the user to copy, edit, and send.

## Motivation
As a Student Assistant (HiWi) at the university, I frequently address email inquiries requesting information that is already available on the website (e.g., application procedures). To manage this, our team uses an Outlook folder containing reusable templates in German and English. These templates typically provide brief explanations and links to detailed online resources.
However, this workflow presents challenges, particularly during the onboarding of new HiWis. The static folder structure can be confusing, and due to the similarity between certain topics, it is often difficult for new team members to determine if a relevant template exists or where to find it. This project aims to optimize this workflow by developing a tool that accepts a raw email as input and suggests the most appropriate template.

## Goals
The primary objective was to build a prototype AI application to streamline the email response process and facilitate the onboarding of new staff.

## Learning Goals:
Evaluate the technical viability of embedding-based retrieval for this specific use case.
Gain practical experience implementing Transformer models with Python.
Improve general Python development skills.

## Preliminary Conclusion
The project confirms that offline sentence-embedded matching is a technically feasible architecture for a Template Suggester, 
but the current prototype requires significant refinement to be operationally reliable.

With a Top-1 accuracy of 63% on the preliminary test set, the system is not yet precise enough for autonomous selection. 
The "score clustering" phenomenon indicates that the current embedding strategy is effective for narrowing 20+ templates 
down to a relevant shortlist, but fails to provide the decisive separation needed for a "one-click" solution. 
To escape this limitation, implementation of hybrid scoring and/or aggressive data engineering (removing boilerplate text from the embedding input) 
is required.

## Project Status
The project was submitted as part of the "KI-Anwendungsentwicklung" module at the Darmstadt University of Applied Sciences. However, development will continue.
