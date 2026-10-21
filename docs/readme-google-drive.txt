===============================================================================
FIDUCIARY LENS - NLP CASE STUDY REVIEW 2
Group 11 | 23CSE471 Natural Language Processing
Amrita Vishwa Vidyapeetham, Coimbatore
===============================================================================

PRODUCT NAME : Fiduciary Lens
SLOGAN       : Every answer, traced back to the law.

-------------------------------------------------------------------------------
LINKS
-------------------------------------------------------------------------------

a. Hosted application URL
   <PASTE THE DEPLOYED URL HERE>

b. GitHub repository
   https://github.com/Key-to-Code/Fiduciary-Lens-Tax-Advisor-System

c. IEEE DataPort dataset
   https://ieee-dataport.org//documents/fiduciary-lens-dataset

d. Demo recording (this folder)
   <PASTE THIS GOOGLE DRIVE FOLDER'S SHARE LINK HERE>

e. Reference journal paper used for experiments and results
   https://www.ijsr.net/getabstract.php?paperid=ES24115141827

-------------------------------------------------------------------------------
CONTENTS OF THIS GOOGLE DRIVE FOLDER
-------------------------------------------------------------------------------

   fiduciary-lens-demo.mp4     Screen recording of the team executing the
                               application and explaining the product
   fiduciary-lens-demo.m4a     Audio track of the same recording
   readme.txt                  This file

-------------------------------------------------------------------------------
TEAM
-------------------------------------------------------------------------------

Roll No              Name                      Contribution
-------------------  ------------------------  ---------------------------------
CB.SC.U4CSE24107     Anuja R S                 Backend APIs and system
                                               integration connecting document
                                               processing, AI modules and the
                                               frontend
CB.SC.U4CSE24118     Ganesh Sarathi            User interface for document
                                               upload, financial summaries and
                                               taxpayer-friendly insights
CB.SC.U4CSE24119     Gaurav Chandrasekhar      AI/NLP pipeline for financial
                                               entity extraction, document
                                               summarisation and tax insight
                                               generation
CB.SC.U4CSE24151     Sudharshan Sambathkumar   Tax-law dataset construction and
                                               preprocessing from official
                                               sources for RAG retrieval

Faculty guide: Dr. Senthil Kumar T., Professor
Department of Computer Science and Engineering
Amrita School of Computing, Coimbatore
t_senthilkumar@cb.amrita.edu

-------------------------------------------------------------------------------
WHAT THE RECORDING SHOWS
-------------------------------------------------------------------------------

00:00  Product introduction and the problem being addressed
01:40  Competitor landscape and the research gap
01:40  LIVE: grounded tax-law question answering with provision-level citations
       - a normal question, with retrieved provisions shown
       - score attribution (XAI) on each retrieved passage
       - refusal of an out-of-scope question, enforced before the LLM is called
       - refusal of a rate question when no Finance Act is indexed
04:10  LIVE: Form 16 document analysis
       - summary, regex and LLM entity extraction
       - per-deduction tax insights tied back to the governing provision
06:10  Architecture, corpus, hybrid retrieval, and the evaluation page
       - retrieval benchmark results
       - SLM vs BERT vs DistilBERT comparison with all metrics
       - occlusion-based XAI attributions
08:10  Results, stated limitations, and deployment links

-------------------------------------------------------------------------------
SYSTEM SUMMARY
-------------------------------------------------------------------------------

Corpus        Income-tax Act 2025, Income-tax Rules 2026, Finance Act 2026
              3,496 passages, chunked at 1,200 characters with 250 overlap

Retrieval     Hybrid: BAAI/bge-small-en-v1.5 dense embeddings in a FAISS index
              fused 65/35 with a BM25 lexical index; deduplicated by provision;
              top 6 passages passed to the generator

Grounding     Refusal enforced in code when the best cosine score falls below
              0.60, checked before the language model is invoked

Generation    meta-llama/llama-3.3-70b-instruct via an OpenAI-compatible
              endpoint, with Ollama, local transformers and a verbatim
              extractive provider as fallbacks

Deep learning SLM (trigram, written from scratch), BERT and DistilBERT
models        compared on statutory-source classification; all performance
              measures are shown in the application's Evaluation page

XAI           Hybrid Retrieval Score Attribution for the retrieval path;
              leave-one-token-out occlusion attribution for the classifiers

-------------------------------------------------------------------------------
SCOPE AND DISCLAIMER
-------------------------------------------------------------------------------

Fiduciary Lens provides educational information about Indian tax law. It is not
professional tax, accounting or financial advice and it is not a substitute for
a Chartered Accountant. It explains what the law says; it does not advise what
to do about an individual return, and it does not compute a tax liability.

===============================================================================
