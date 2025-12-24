# Copyright (c) 2025 Marco Fago
#
# This code is licensed under the MIT License.
# See the LICENSE file in the repository for the full license text.

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# For better security, load environment variables from a .env file
from dotenv import load_dotenv

load_dotenv()
# Make sure your GROQ_API_KEY is set in the .env file
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# --- Prompt 1: Extract Information ---
prompt_extract = ChatPromptTemplate.from_template(
"Extract the technical specifications from the following text:\n\n{text_input}"
)
# --- Prompt 2: Transform to JSON ---
prompt_transform = ChatPromptTemplate.from_template(
"Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
)
# --- Build the Chain using LCEL ---
# The StrOutputParser() converts the LLM's message output to a simple string.
extraction_chain = prompt_extract | llm | StrOutputParser()
# The full chain passes the output of the extraction chain into the 'specifications'
# variable for the transformation prompt.
full_chain = (
{"specifications": extraction_chain}
| prompt_transform
| llm
| StrOutputParser()
)
# --- Run the Chain ---
input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."
# Execute the chain with the input text dictionary.
final_result = full_chain.invoke({"text_input": input_text})
print("\n--- Final JSON Output ---")
print(final_result)