from transformers import pipeline


def test(prompt, model='facebook/opt-125m'):
    generator = pipeline('text-generation', model)
    output = generator(prompt, max_length=50, do_sample=True)
    print(f"Modelo {model}: {output[0]['generated_text']}")


if __name__ == '__main__':
    prompt = "O Ministério Público Federal em Santa Catarina investiga a construção "
    for _ in range(10):
        test(prompt)
        test(prompt, "1024_tokens_4_bs/opt-PT/checkpoint-303861")
