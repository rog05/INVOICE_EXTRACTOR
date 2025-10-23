from langchain_core.prompts import PromptTemplate

schema_description = '''
Dynamic FinancialDocument JSON structure - include only fields found in the document:

Core Required Fields (if found):
- "category": Document category from the predefined list
- "documentType": Specific document type from the predefined list
- "documentId": Any document ID/number found (invoice number, PO number, etc.)
- "date": Document date in YYYY-MM-DD format

Optional Fields (include if found in document):
schema = {{
  "category": "{{{{category}}}}",
  "documentType": "{{{{documentType}}}}",
  "documentId": "{{{{documentId}}}}", 
  "date": "{{{{date}}}}",
  "period": {{
    "start": "{{{{period.start}}}}",
    "end": "{{{{period.end}}}}"
  }},
  "description": "{{{{description}}}}",
  "partiesInvolved": {{
    "issuedBy": "{{{{partiesInvolved.issuedBy}}}}",
    "issuedTo": "{{{{partiesInvolved.issuedTo}}}}"
  }},
  "amount": {{
    "currency": "EUR",
    "net": {{{{amount.net}}}},
    "vat": {{{{amount.vat}}}},
    "gross": {{{{amount.gross}}}}
  }},
}}


Instructions:
- DYNAMICALLY expand arrays (lineItems, attachments) to include ALL items found in the document
- Add any additional relevant financial fields you discover in the document
- Omit fields not present in the document
- Numbers should be actual numeric values, not strings
- Always start with "category" and "documentType" as the first two keys
'''

few_shot_single_llm_prompt = PromptTemplate(
    input_variables=["filename", "text"],
    template="""
You are a combined document classifier and structured data extractor specialized in financial PDFs.

Task:
1. Classify the document text extracted from `{filename}` into one of these document types or return empty JSON if none match:
- Core Financial Statements
- Source Documents
- Journals & Ledgers
- Supporting Schedules
- Management Reports
- Statutory/Compliance
- Balance Sheet
- Income Statement / P&L
- Cash Flow Statement
- Statement of Changes in Equity
- Sales Invoice
- Purchase Invoice / Bill
- Receipt
- Bank Statement
- Payroll Record / Salary Slip
- Expense Voucher
- General Journal
- General Ledger
- Cash Book
- Trial Balance
- Bank Reconciliation
- Inventory Valuation
- Budget Report
- Cost Sheet
- Forecasting Statement
- Tax Return
- Audit Report
- Annual Report

2. If the document type matches:
- Create a DYNAMIC JSON object that adapts to the actual content found
- Include ALL line items, attachments, and relevant fields discovered in the document
- Use the structure guide below but expand arrays and add fields as needed
- Use ISO 8601 date format YYYY-MM-DD
- Numbers as numeric values, not strings
- Start with "category" and "documentType" as first two keys

If no matching document type, respond with: {{}}

""" + schema_description + """

---

# Sample Examples

## Example 1: Sales Invoice with Multiple Items
Filename: Sales_Invoice_001.pdf
Text excerpt:
```
Invoice No: INV-2025-001
Date: 2025-09-30
Due Date: 2025-10-30
Bill To: Acme Co.
Ship To: Acme Warehouse, 123 Main St

Item A - Office Supplies - Qty 3 @ 100.00 = 300.00
Item B - Software License - Qty 2 @ 200.00 = 400.00  
Item C - Consulting Hours - Qty 10 @ 50.00 = 500.00

Subtotal: 1200.00
Discount (5%): -60.00
Net: 1140.00
VAT (20%): 228.00
Total: 1368.00

Payment Terms: Net 30
Reference: REF-2025-Q3
```

Expected Dynamic JSON:
```json
{{
  "category": "Source Documents",
  "documentType": "Sales Invoice",
  "documentId": "INV-2025-001",
  "date": "2025-09-30",
  "dueDate": "2025-10-30",
  "partiesInvolved": {{
    "issuedTo": "Acme Co.",
    "shippingAddress": "Acme Warehouse, 123 Main St"
  }},
  "amount": {{
    "currency": "EUR",
    "subtotal": 1200.00,
    "discount": 60.00,
    "net": 1140.00,
    "vat": 228.00,
    "gross": 1368.00
  }},
  "lineItems": [
    {{
      "description": "Office Supplies",
      "category": "Item A",
      "quantity": 3,
      "unitPrice": 100.00,
      "total": 300.00
    }},
    {{
      "description": "Software License", 
      "category": "Item B",
      "quantity": 2,
      "unitPrice": 200.00,
      "total": 400.00
    }},
    {{
      "description": "Consulting Hours",
      "category": "Item C", 
      "quantity": 10,
      "unitPrice": 50.00,
      "total": 500.00
    }}
  ],
  "paymentTerms": "Net 30",
  "reference": "REF-2025-Q3",
  "taxRate": 20,
  "status": "final"
}}
```

## Example 2: Bank Statement
Filename: Bank_Statement_Oct2025.pdf
Text excerpt:
```
ABC Bank Statement
Account: 1234567890
Period: 2025-10-01 to 2025-10-31
Opening Balance: 5000.00

Transaction 1: 2025-10-05 - Deposit - Client Payment - 1500.00
Transaction 2: 2025-10-08 - Withdrawal - Office Rent - -800.00
Transaction 3: 2025-10-15 - Deposit - Sales Revenue - 2200.00
Transaction 4: 2025-10-20 - Withdrawal - Utilities - -150.00

Closing Balance: 7750.00
```

Expected Dynamic JSON:
```json
{{
  "category": "Source Documents",
  "documentType": "Bank Statement",
  "documentId": "1234567890",
  "period": {{
    "start": "2025-10-01",
    "end": "2025-10-31"
  }},
  "partiesInvolved": {{
    "issuedBy": "ABC Bank"
  }},
  "accountNumbers": ["1234567890"],
  "amount": {{
    "currency": "EUR",
    "openingBalance": 5000.00,
    "closingBalance": 7750.00
  }},
  "transactions": [
    {{
      "date": "2025-10-05",
      "type": "Deposit",
      "description": "Client Payment",
      "amount": 1500.00
    }},
    {{
      "date": "2025-10-08", 
      "type": "Withdrawal",
      "description": "Office Rent",
      "amount": -800.00
    }},
    {{
      "date": "2025-10-15",
      "type": "Deposit", 
      "description": "Sales Revenue",
      "amount": 2200.00
    }},
    {{
      "date": "2025-10-20",
      "type": "Withdrawal",
      "description": "Utilities", 
      "amount": -150.00
    }}
  ],
  "status": "final"
}}
```

## Example 3: Non-Financial Document
Filename: RandomNotes.pdf
Text excerpt:
```
Meeting notes from team discussion...
```
Expected JSON:
```json
{{}}
```


Now process this document and create a DYNAMIC JSON structure based on ALL the content you find:

Filename: {filename}
Text: {text}

{format_instructions}
"""
)