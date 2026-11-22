import ollama

user_input = input("How do you want to edit this image? \n")

def rewrite(user_input):
    templated_input = f"""You are an instruction translator for an InstructPix2Pix image-editing model.

Your task is to convert natural-language image-editing requests into short, direct, literal editing instructions.

Follow these rules:

1. Preserve the user's original intent. Do not add edits that the user did not request or reasonably imply.
2. Use simple, concrete commands such as "Add", "Remove", "Change", "Replace", "Make", and "Turn".
3. Describe visual changes rather than subjective feelings or abstract concepts.
4. When the user requests multiple edits, preserve every requested edit while keeping the instruction concise.
5. Do not describe the entire image. Only describe the requested changes.
6. Do not introduce new objects, colors, people, backgrounds, or visual details without justification from the original request.
7. Preserve aspects of the image that the user did not ask to change.
8. Resolve vague language conservatively. If the intended edit cannot be determined, do not invent a specific interpretation.
9. Keep the output below 77 CLIP tokens.
10. Return only the rewritten editing instruction. Do not provide explanations, alternatives, or commentary.

Original request: {user_input}

Rewritten instruction:"""
    # f"rewrite the prompt '{user_input}' in a way that follows these instructions: InstructPix2Pix only knows instructions in the style it was trained on: short, literal commands written by GPT-3, like “make it snowy” or “turn the dog into a cat”. Real users don’t write that way. They’re vague (“make it pop”) or indirect (“it should feel like fall”), or they ask for several changes at once, and the CLIP text encoder reads only 77 tokens. When an instruction doesn’t look like the training data, the model guesses. We want to close that gap between how people ask and what the model understands. One way is to put a language model in front of the editor that translates what the user typed into the kind of instruction the model was trained on, without changing what they meant. Also change the prompt to emphasize not changing anything other than what the prompt wants you to chang (like if they want a persons hair to chanhe to brown, make sure the prompt emphasizes that only the hair color should change, and nothing else)" 
    # artifical intelligece editors can understand properly using specific language, and make this new prompt under 25 tokens. Do not change anything else about the image except the what it wants you to change. For example, if it says to change the color of the couches to blue, only change the color of the couches and nothing else. Keep the detail and everything, and make sure to EDIT THE PROMPT IN A WAY THAT TELLS THE AI TO ONLY MAKE THE CHANGES THAT THE ORIGINAL PROMPY CONVEYED - don't add any fluff, just make the prompt and dont send back anything else"

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[
            {
                "role": "user", 
                "content": templated_input 
            }
        ],
    )

    ai_output = response["message"]["content"]
    return ai_output

#print(ai_output)

