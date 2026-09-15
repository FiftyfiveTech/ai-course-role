# Seed Licences and Provenance

20 scenario seeds — 8 dev / 12 heldout — fetched from three public HF datasets.
No bulk download: each row fetched individually via the HF dataset viewer API.

| Seed ID | Dataset (HF repo id) | Row index | Licence | Split |
|---------|----------------------|-----------|---------|-------|
| soda-0100 | allenai/soda | 100 | Apache-2.0 | dev |
| soda-0500 | allenai/soda | 500 | Apache-2.0 | dev |
| soda-1000 | allenai/soda | 1000 | Apache-2.0 | dev |
| hh-80002 | Anthropic/hh-rlhf | 80002 | MIT | dev |
| hh-80012 | Anthropic/hh-rlhf | 80012 | MIT | dev |
| hh-80019 | Anthropic/hh-rlhf | 80019 | MIT | dev |
| bitext-1000 | bitext/Bitext-customer-support-llm-chatbot-training-dataset | 1000 | CC-BY-4.0 | dev |
| bitext-5000 | bitext/Bitext-customer-support-llm-chatbot-training-dataset | 5000 | CC-BY-4.0 | dev |
| soda-0502 | allenai/soda | 502 | Apache-2.0 | heldout |
| soda-1500 | allenai/soda | 1500 | Apache-2.0 | heldout |
| soda-2000 | allenai/soda | 2000 | Apache-2.0 | heldout |
| soda-2500 | allenai/soda | 2500 | Apache-2.0 | heldout |
| hh-80007 | Anthropic/hh-rlhf | 80007 | MIT | heldout |
| hh-80014 | Anthropic/hh-rlhf | 80014 | MIT | heldout |
| hh-80015 | Anthropic/hh-rlhf | 80015 | MIT | heldout |
| hh-80023 | Anthropic/hh-rlhf | 80023 | MIT | heldout |
| bitext-3000 | bitext/Bitext-customer-support-llm-chatbot-training-dataset | 3000 | CC-BY-4.0 | heldout |
| bitext-7000 | bitext/Bitext-customer-support-llm-chatbot-training-dataset | 7000 | CC-BY-4.0 | heldout |
| bitext-10000 | bitext/Bitext-customer-support-llm-chatbot-training-dataset | 10000 | CC-BY-4.0 | heldout |
| bitext-12000 | bitext/Bitext-customer-support-llm-chatbot-training-dataset | 12000 | CC-BY-4.0 | heldout |

## Dataset cards and licence texts

### allenai/soda — Apache-2.0
<https://huggingface.co/datasets/allenai/soda>

Social dialogue dataset derived from ATOMIC commonsense knowledge. Licence: Apache 2.0.

### Anthropic/hh-rlhf — MIT
<https://huggingface.co/datasets/Anthropic/hh-rlhf>

Human preference data for helpful and harmless RLHF training.
Rows selected from the high-offset range (80000+) which contains the
helpful-base split — general-knowledge Q&A dialogues.
Licence: MIT.

### bitext/Bitext-customer-support-llm-chatbot-training-dataset — CC-BY-4.0
<https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset>

Structured customer support dialogues labelled by intent and category.
Six distinct intents selected (change_order, check_cancellation_fee,
check_payment_methods, complaint, create_account, delivery_options).
Licence: Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Fetch method

Each row was fetched via the HF dataset viewer REST API:

```
GET https://datasets-server.huggingface.co/rows
    ?dataset=<hf-repo-id>&config=default&split=train&offset=<row_index>&length=1
```

The fetch script is at `scripts/fetch_seeds.py` and is idempotent.
