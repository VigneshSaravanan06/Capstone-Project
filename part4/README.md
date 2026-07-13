# Part 4: LLM-Powered Feature – Tabular Record Batch Scoring (Track B)

## Project Overview

This project adds a Large Language Model (LLM) feature to the machine learning project completed in Parts 1–3.

The chosen feature is **Track B: Tabular Record Batch Scoring**.

The application reads patient records from **cleaned_data.csv**, sends them to an LLM, receives a structured JSON response, validates the response using a JSON schema, checks for PII (email and phone numbers) before sending data, and compares the outputs using different temperature settings.

---

## Chosen Track

**Track B – Tabular Record Batch Scoring**

Patient records from **cleaned_data.csv** were sent to the LLM individually.

For each record, the LLM returned:

* Risk Tier
* Flag for Review
* Primary Signal
* Confidence
* Recommended Action

Each response was validated using a JSON schema.

---

## Dataset

The project uses the cleaned dataset created in Part 1.

**Dataset:** `cleaned_data.csv`

The dataset contains demographic, medical, clinical, and lifestyle information used for heart attack risk assessment.

---

## LLM API Connection

The project uses the **OpenRouter Chat Completions API** with the Python **requests** library.

The API key is securely loaded from Google Colab Secrets using:

`os.environ["LLM_API_KEY"]`

This prevents the API key from being exposed in the notebook.

---

## Reusable LLM Function

A reusable function called **call_llm()** was created.

The function:

* Creates the API request
* Sends the request
* Handles errors
* Reads the response
* Returns the assistant's reply

The function was tested successfully using a simple prompt.

---

## Prompt Design

### System Prompt

The system prompt instructs the LLM to:

* Assess the patient record
* Return only valid JSON
* Avoid Markdown
* Avoid extra explanations

It also includes one example input and output.

### User Prompt

Each patient record is inserted into the following template:

```text
Assess the following de-identified patient record.

Return only the required JSON object.

Input Record:
{record_json}
```

---

## Temperature Comparison

Each patient record was evaluated using two temperature settings:

* Temperature = 0
* Temperature = 0.7

### Why Use Temperature = 0?

Temperature **0** produces consistent and repeatable JSON responses, making it suitable for structured tasks.

Temperature **0.7** introduces more variation in wording and confidence values because the model generates more diverse responses.

---

## JSON Schema Validation

Each response contains the following required fields:

* risk_tier
* flag_for_review
* primary_signal
* confidence
* recommended_action

Responses are parsed using `json.loads()` and validated using `jsonschema.validate()` before further processing.

---

## Batch Scoring

The notebook processes patient records one by one and displays:

* Input record
* LLM response
* Parsed JSON
* Validation result

---

## PII Guardrail

Before sending data to the LLM, the application checks for:

* Email addresses
* Phone numbers

If PII is detected, the request is blocked to prevent sensitive information from being sent to the API.

---

## Guardrail Test

The guardrail was tested with inputs containing and not containing PII to verify that it correctly blocks sensitive information while allowing safe inputs.

---

## End-to-End Demonstration

The notebook demonstrates the complete workflow:

* Reading patient records
* Sending requests to the LLM
* Receiving structured JSON
* Validating the JSON
* Applying the PII guardrail
* Saving the final results

---

## Generated Files

```
results/
├── batch_scoring_results.csv
├── guardrail_results.csv
├── temperature_comparison.csv
├── temperature_validation_results.csv
└── final_demonstration_table.csv
```

Project files:

```
part4/
├── Heart_Attack_Part4.ipynb
├── README.md
├── requirements.txt
├── cleaned_data.csv
├── .env.example
└── .gitignore
```

---

## How to Run

1. Open `Heart_Attack_Part4.ipynb` in Google Colab.
2. Add your OpenRouter API key as **LLM_API_KEY** in Colab Secrets.
3. Upload `cleaned_data.csv`.
4. Run all notebook cells.
5. Verify that:

   * The API connection succeeds.
   * Patient records are scored.
   * JSON validation succeeds.
   * The PII guardrail works correctly.
   * Result CSV files are generated.

---

## Dependencies

* pandas
* requests
* jsonschema
* python-dotenv (optional)

---

## Limitations

* The LLM assessments are for educational purposes only and should not be used for medical diagnosis.
* Results may vary depending on the LLM model.
* Higher temperature values may produce different responses.
* The PII guardrail detects common email and phone number formats but may not detect every form of sensitive information.

---

## Conclusion

This project demonstrates an LLM-powered patient record scoring pipeline using structured prompts, JSON schema validation, and PII guardrails. The implementation provides consistent structured outputs and shows that **temperature = 0** is the most reliable setting for automated scoring tasks.

