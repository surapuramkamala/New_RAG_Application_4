Architecture
                    USER
                      │
                      ▼
                FastAPI API
                      │
                      ▼
              Query Transformation
                  (Gemini)
                      │
                      ▼
          ┌─────────────────────┐
          │   HYBRID RETRIEVAL  │
          ├─────────────────────┤
          │                     │
          ▼                     ▼
    Vector Search          Keyword Search
       FAISS                  BM25
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
                 Reranking
                     │
                     ▼
              Metadata Filter
                     │
                     ▼
               Top Documents
                     │
                     ▼
             Gemini Answer LLM
                     │
                     ▼
       Answer + Citations + Confidence


1. access_matrix.csv
What access level does a Developer have for the Production_Server?
Which role has Admin access to the Production_Server?
What approval is required for a Senior Developer to access the Production Server?
Which resources require MFA, and which roles can access them?
Compare the Production Server access permissions for Developer, Senior Developer, QA, and DevOps.

2. Contract_Template_for_Consulting Services Agreement 1.docx
What is the purpose of the Consulting Services Agreement?
What are the responsibilities of the Key Personnel under the agreement?
How and when is the Consultancy Fee paid?
What confidentiality obligations does the Consultant have?
What restrictions does the agreement place on the Consultant regarding working with competitors?

3. employee_policy.pdf
What is the purpose of the Acceptable Use Policy?
What types of personal use of IT resources are allowed for employees?
Give examples of activities considered unacceptable use of IT resources.
What are the requirements for handling Controlled Unclassified Information (CUI)?
What happens when a user violates the IT or cybersecurity policies?

4. incident_process.txt
What are the main stages in the security incident-response lifecycle?
What factors should the security team consider during the initial assessment of an incident?
What containment actions can be performed when an incident presents an immediate security risk?
What activities are included in the eradication phase?
What information should be documented throughout the incident-response lifecycle?

5. production_access.pdf
Who is authorized to access non-public Information Resources?
What information must be documented when creating a new account?
When should accounts be disabled for users who leave the organization?
What security requirements apply to remote and wireless access?
What are the requirements for third-party access to University Data?

6. redrafted_contract_20260413_155405.pdf
What is the aggregate liability cap proposed for each SOW?
Which liabilities are excluded from the general liability cap?
What data protection and information security responsibilities does the Supplier have?
How quickly must the Supplier notify G42 about a data breach or security incident?
What rights does G42 have regarding audits, data return, and data deletion?