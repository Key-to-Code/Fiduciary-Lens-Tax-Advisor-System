import json

def generate_chat_intent_dataset(output_json_path="chat_intent_testset.json"):
    # High-yield user evaluation scenarios map
    intents_data = [
        {
            "intent_id": "INTENT_001",
            "category": "HRA_Exemption",
            "sample_queries": [
                "Can I claim rent paid to my parents?",
                "My landlord is my dad, can I get HRA exemption?",
                "Is it legal to show rent receipts to parents to save tax?"
            ],
            "expected_ground_truth_reference": "Section 10(13A)",
            "required_entities": ["rent_amount", "landlord_pan"]
        },
        {
            "intent_id": "INTENT_002",
            "category": "Section_80D_Medical",
            "sample_queries": [
                "What is the maximum limit for health insurance tax deduction?",
                "How much can I claim for medical insurance under 80D?",
                "Can I claim my parents' senior citizen health checkup?"
            ],
            "expected_ground_truth_reference": "Section 80D",
            "required_entities": ["insurance_premium", "parent_age_bracket"]
        },
        {
            "intent_id": "INTENT_003",
            "category": "Capital_Gains_Mutual_Funds",
            "sample_queries": [
                "Is my mutual fund long term or short term?",
                "What is the tax rate on equity mutual fund profit?",
                "How is short term capital gains calculated on stocks?"
            ],
            "expected_ground_truth_reference": "Section 111A / Section 112A",
            "required_entities": ["holding_period", "asset_type"]
        }
    ]
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(intents_data, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully exported {len(intents_data)} chat evaluation categories to '{output_json_path}'.")
    print("You can use this file as the validation golden dataset for your automated testing loops.")

if __name__ == "__main__":
    generate_chat_intent_dataset()