from transformers import pipeline

# Load a pre-trained language generation model
generator = pipeline('text-generation', model='gpt2')

# Generate an answer based on the retrieved context
def generate_answer(context, question):
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    result = generator(input_text, max_length=50)
    return result[0]['generated_text']

# Save the underlying model and tokenizer to a folder
generator.model.save_pretrained("./gpt2_saved")
generator.tokenizer.save_pretrained("./gpt2_saved")
